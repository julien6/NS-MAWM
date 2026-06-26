
benchmarl.algorithms
====================

.. currentmodule:: benchmarl.algorithms

.. contents:: Contents
    :local:

Here you can find the :ref:`algorithm table <algorithm-table>`.

Common
------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class_private.rst

   Algorithm
   AlgorithmConfig

Algorithms
----------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class_private.rst

   {% for name in benchmarl.algorithms.classes %}
     {{ name }}
   {% endfor %}

MB-MAPPO
--------

``MBMappo`` is an experimental model-based variant of MAPPO. It keeps the
actor update on-policy by default: PPO actor losses are computed only from
real trajectories collected by the BenchMARL collector. A supervised MLP world
model is trained online from those real transitions and short imagined
rollouts are used only to improve critic value targets and advantages.

The first implementation is intentionally minimal. The world model predicts
local next observations, rewards, and optionally done logits. Imagined rollouts
are short and are not inserted into the PPO replay buffer unless
``algorithm.imagined_rollouts.use_for_actor`` is explicitly enabled, which is
experimental and breaks the strict on-policy interpretation.

Example:

.. code-block:: bash

   python benchmarl/run.py algorithm=mb_mappo task=vmas/balance
   python benchmarl/run.py algorithm=mb_mappo task=vmas/balance algorithm.imagined_rollouts.horizon=5 algorithm.imagined_rollouts.num_branches=8
