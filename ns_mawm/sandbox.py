"""Bounded interpreter for the rule subset; generated source never reaches exec/eval."""
from __future__ import annotations
import ast
import operator
from types import MappingProxyType
from collections.abc import Mapping


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class Interpreter:
    def __init__(self, safe_calls, safe_attrs, budget=20000):
        self.calls, self.attrs, self.remaining = safe_calls, safe_attrs, budget

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise TimeoutError("Rule instruction budget exceeded")

    def run(self, function, context):
        scope = {function.args.args[0].arg: context, **self.calls}
        try:
            self.statements(function.body, scope)
        except Returned as result:
            return result.value
        return None

    def assign(self, target, value, scope):
        self.tick()
        if isinstance(target, ast.Name):
            scope[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
            for t, v in zip(target.elts, value):
                self.assign(t, v, scope)
        else:
            raise ValueError("Only local variable assignments are allowed")

    def statements(self, nodes, scope):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, scope) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, scope)
                for target in node.targets:
                    self.assign(target, value, scope)
            elif isinstance(node, ast.AnnAssign):
                self.assign(node.target, self.expr(node.value, scope), scope)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expr(node.test, scope) else node.orelse, scope)
            elif isinstance(node, ast.For):
                for value in self.expr(node.iter, scope):
                    self.assign(node.target, value, scope)
                    self.statements(node.body, scope)
                self.statements(node.orelse, scope)
            elif isinstance(node, ast.Pass):
                pass
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                pass  # docstring
            elif not isinstance(node, (ast.Return, ast.Assign)):
                raise ValueError(f"Unsupported rule statement: {type(node).__name__}")

    def comprehension(self, node, scope):
        result = []
        def walk(index, local):
            self.tick()
            if index == len(node.generators):
                result.append((self.expr(node.key, local), self.expr(node.value, local)) if isinstance(node, ast.DictComp) else self.expr(node.elt, local))
                return
            generator = node.generators[index]
            for v in self.expr(generator.iter, local):
                inner = dict(local)
                self.assign(generator.target, v, inner)
                if all(self.expr(condition, inner) for condition in generator.ifs):
                    walk(index+1, inner)
        walk(0, dict(scope))
        return dict(result) if isinstance(node, ast.DictComp) else (set(result) if isinstance(node, ast.SetComp) else result)

    def expr(self, node, scope):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.JoinedStr):
            value = "".join(str(self.expr(part, scope)) for part in node.values)
            if len(value) > 10000:
                raise ValueError("Formatted string allocation limit exceeded")
            return value
        if isinstance(node, ast.FormattedValue):
            if node.format_spec is not None:
                raise ValueError("Format specifications are not supported")
            return str(self.expr(node.value, scope))
        if isinstance(node, ast.Name):
            return scope[node.id]
        if isinstance(node, ast.Attribute):
            if node.attr not in self.attrs:
                raise ValueError("Attribute not allowed")
            return getattr(self.expr(node.value, scope), node.attr)
        if isinstance(node, ast.Subscript):
            return self.expr(node.value, scope)[self.expr(node.slice, scope)]
        if isinstance(node, ast.Slice):
            return slice(*(self.expr(n, scope) if n else None for n in (node.lower, node.upper, node.step)))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            items = [self.expr(n, scope) for n in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else (set(items) if isinstance(node, ast.Set) else items)
        if isinstance(node, ast.Dict):
            return {self.expr(k, scope): self.expr(v, scope) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            return self.comprehension(node, scope)
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, scope) else node.orelse, scope)
        if isinstance(node, ast.BoolOp):
            value = None
            for n in node.values:
                value = self.expr(n, scope)
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
            return value
        if isinstance(node, ast.UnaryOp):
            ops = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
            return ops[type(node.op)](self.expr(node.operand, scope))
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, scope), self.expr(node.right, scope)
            if isinstance(node.op, ast.Add) and isinstance(left, (str, list, tuple)):
                if len(left) + len(right) > 10000:
                    raise ValueError("Sequence allocation limit exceeded")
                return left + right
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError("Only numeric arithmetic and bounded concatenation are allowed")
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            value = ops[type(node.op)](left, right)
            if isinstance(value, int) and value.bit_length() > 1024:
                raise ValueError("Integer arithmetic limit exceeded")
            return value
        if isinstance(node, ast.Compare):
            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                   ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not,
                   ast.In: lambda a,b: a in b, ast.NotIn: lambda a,b: a not in b}
            left = self.expr(node.left, scope)
            for op, next_node in zip(node.ops, node.comparators):
                right = self.expr(next_node, scope)
                if not ops[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            # Resolve by syntactic name from the immutable whitelist, never from
            # local variables (which might shadow a whitelisted function name).
            if isinstance(node.func, ast.Name):
                function = self.calls[node.func.id]
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {"get", "items", "keys", "values"}:
                base = self.expr(node.func.value, scope)
                if not isinstance(base, Mapping):
                    raise ValueError("Only immutable mapping methods are callable")
                function = getattr(base, node.func.attr)
            else:
                raise ValueError("Indirect call forbidden")
            return function(*[self.expr(a, scope) for a in node.args], **{k.arg: self.expr(k.value, scope) for k in node.keywords})
        raise ValueError(f"Unsupported rule expression: {type(node).__name__}")


def bind(function, safe_calls, safe_attrs):
    if len(function.args.args) != 1 or function.args.defaults or function.decorator_list or function.args.vararg or function.args.kwarg:
        raise ValueError("Rule functions take exactly one context and no defaults or decorators")
    def invoke(context):
        return Interpreter(safe_calls, safe_attrs).run(function, context)
    return invoke
