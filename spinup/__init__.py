# Disable TF deprecation warnings.
# Syntax from tf1 is not expected to be compatible with tf2.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# # Algorithms

from spinup.algos.pytorch.dqn.dqn import dqn as dqn_pytorch
from spinup.algos.pytorch.dqn.exploit import exploit as exploit_pytorch
# from spinup.algos.pytorch.tabsarsa.tabsarsa import tabsarsa as tabsarsa_pytorch
# from spinup.algos.pytorch.tabsarsa.tabsarsa import tabsarsa_store_q_tables as tabsarsa_store_q_tables_pytorch
# from spinup.algos.pytorch.tabsarsa.tabsarsa import store_q_table as store_q_table_pytorch
# from spinup.algos.pytorch.tabsarsa.tabsarsa import store_vi_q_table as store_vi_q_table_pytorch
from spinup.algos.pytorch.tabsarsa.tabsarsa import test_xis_for_q_star as test_xis_for_q_star_pytorch

# Loggers
from spinup.utils.logx import Logger, EpochLogger

# Version
from spinup.version import __version__