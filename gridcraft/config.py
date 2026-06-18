from collections import namedtuple

from exp_config import ACTION_SIZE, CONTROLLER_INPUT_SIZE

Game = namedtuple('Game', ['env_name', 'input_size', 'output_size', 'activation'])

games = {}

gridcraft = Game(
  env_name='gridcraft',
  input_size=CONTROLLER_INPUT_SIZE,
  output_size=ACTION_SIZE,
  activation='passthru',
)
games['gridcraft'] = gridcraft

gridcraftdream = Game(
  env_name='gridcraftdream',
  input_size=CONTROLLER_INPUT_SIZE,
  output_size=ACTION_SIZE,
  activation='passthru',
)
games['gridcraftdream'] = gridcraftdream
