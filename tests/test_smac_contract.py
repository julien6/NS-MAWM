import torch
from ns_mawm.smac import codec, encode
from ns_mawm.libraries import smacv2
from ns_mawm.rules import Context, PSWM


def features():
    names=['move_action_north','move_action_south','move_action_east','move_action_west']
    names+=['enemy_shootable_0','enemy_distance_0','enemy_relative_x_0','enemy_relative_y_0','enemy_health_0','enemy_unit_type_0_bit_0','enemy_unit_type_0_bit_1']
    names+=['ally_visible_1','ally_distance_1','ally_relative_x_1','ally_relative_y_1','ally_health_1','ally_unit_type_1_bit_0','ally_unit_type_1_bit_1']
    names+=['own_health','own_pos_x','own_pos_y','own_unit_type_bit_0','own_unit_type_bit_1']
    return names


def test_smac_dead_and_categories():
    names=features()
    schema,mappings=codec(names,2)
    obs=encode(schema,mappings,torch.zeros(2,len(names)))
    decoded=schema.decode(obs)
    assert decoded['agent0.own_unit_type']=='unobserved'
    assert 'agent1.ally_visible_0' in decoded
    lib=smacv2(schema,2)
    output=PSWM(lib)(Context(decoded,(0,0)))
    assert torch.equal(output.target,obs)
    assert output.mask.all()
    assert any(r.scope=='joint' for r in lib.rules)


def test_smac_live_visibility_type_and_health():
    names=features()
    schema,mappings=codec(names,2)
    native=torch.zeros(2,len(names))
    for name,value in {'own_health':1.,'own_unit_type_bit_0':1.,'enemy_distance_0':.3,'enemy_health_0':.8,
                       'ally_visible_1':1.,'ally_distance_1':.2,'ally_health_1':1.}.items():
        native[:,names.index(name)]=value
    decoded=schema.decode(encode(schema,mappings,native))
    output=PSWM(smacv2(schema,2))(Context(decoded,(1,1)))
    assert 'agent0.own_unit_type' in output.provenance
    assert output.provenance['agent0.enemy_health_0']
    assert output.provenance['agent1.ally_visible_0']
