from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np

from connect4_zero.agent.api_connect4 import Connect4ModelAPI
from connect4_zero.config import Config
from connect4_zero.env.connect4_env import Connect4Env, Winner, Player

CounterKey = namedtuple("CounterKey", "board next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

logger = getLogger(__name__)


class MctsNode:
    def __init__(self, p, v, action_map, is_terminal=False):
        if is_terminal:
            self.Z = v
            self.V = v
            return
        n = len(p)
        if n == 0:
            self.Z = 0
            self.V = 0
            return
        self.P = np.array(p)
        self.V = v
        self.action_map = action_map
        self.N = np.zeros([n,])
        self.W = np.zeros([n, ])
        self.Q = np.zeros([n, ])
        self.N = np.zeros([n, ])
        self.children = [None] * n
        self.Z = None
        self.sem = asyncio.Semaphore()

    def is_leaf(self):
        return self.Z is not None

    def is_terminal(self):
        return self.Z is not None

    def reward(self):
        return self.Z

    async def select(self, is_root, config):
        if self.is_leaf():
            return None, None
        with await self.sem:
            action_t = None
            for i, node in enumerate(self.children):
                if node is not None and node.reward() == -1:
                    action_t = i
                    break
            if action_t is None:
                xx_ = np.sum(self.N)  # SQRT of sum(N(s, b); for all b)
                xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
                xx_ = np.sqrt(xx_)
                p_ = self.P

                if is_root:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
                    p_ = (1 - config.noise_eps) * p_ + config.noise_eps * np.random.dirichlet([config.dirichlet_alpha] * len(p_))

                u_ = config.c_puct * p_ * xx_ / (1 + self.N)
                v_ = (self.Q + u_ + 1000)
                # noinspection PyTypeChecker
                action_t = int(np.argmax(v_))
            virtual_loss = config.virtual_loss
            n = self.N[action_t] = self.N[action_t] + virtual_loss
            w = self.W[action_t] = self.W[action_t] - virtual_loss
            self.Q[action_t] = w / n
            return action_t, self.children[action_t]

    async def backup(self, v, action_t, config):
        with await self.sem:
            n = self.N[action_t] = self.N[action_t] - config.virtual_loss + 1
            w = self.W[action_t] = self.W[action_t] + config.virtual_loss + v
            self.Q[action_t] = w / n

class Connect4Player:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = Connect4ModelAPI(self.config, self.model)

        self.labels_n = config.n_labels
        self.var_nodes = {}
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, board):

        env = Connect4Env().update(board)
        key = self.counter_key(env)
        pc = self.play_config

        for tl in range(self.play_config.thinking_loop):
            if tl > 0 and self.play_config.logging_thinking:
                logger.debug(f"continue thinking: policy : {action + 1} N={node.N} p={node.P}")
                logger.debug(f"value : {action_by_value + 1} : {node.Q}")

            self.search_moves(board)
            node = self.var_nodes[key]
            if self.play_config.logging_thinking:
                logger.debug(f"N={node.N} p={node.P} Q={node.Q}")
            if env.turn < pc.change_tau_turn:
                policy_t = node.N / np.sum(node.N)  # tau = 1
                policy = np.zeros([self.labels_n, ])
                for i in range(len(node.action_map)):
                    policy[node.action_map[i]] = policy_t[i]
                action = int(np.random.choice(range(self.labels_n), p=policy))
                break
            else:
                action = np.argmax(node.N)  # tau = 0
                action_by_value = int(np.argmax(node.Q + (node.N > 0) * 100))
                if action == action_by_value or tl == self.play_config.thinking_loop - 1:
                    policy = np.zeros([self.labels_n,])
                    policy[action] = 1
                    action = node.action_map[action]
                    break

        # this is for play_gui, not necessary when training.
        node_q = np.zeros([self.labels_n, ])
        node_n = np.zeros([self.labels_n, ])
        for i, act in enumerate(node.action_map):
            node_q[act] = node.Q[i]
            node_n[act] = node.N[i]
        self.thinking_history[env.observation] = HistoryItem(action, policy, list(node_q), list(node_n))

        self.moves.append([env.observation, list(policy)])
        return action

    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    def search_moves(self, board):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(board)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, board):
        self.running_simulation_num += 1
        env = Connect4Env().update(board)
        node, _ = await self.create_node(env)

        with await self.sem:  # reduce parallel search number
            env = Connect4Env().update(board)
            leaf_v = await self.search_my_move(env, node)
            self.running_simulation_num -= 1
            return leaf_v

    async def create_node(self, env:Connect4Env) -> MctsNode:
        key = self.counter_key(env)
        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)
        if key in self.var_nodes:
            return self.var_nodes[key], False
        self.now_expanding.add(key)
        if env.done:
            if env.winner == Winner.draw:
                leaf_v = 0
            elif env.winner == env.player_turn():
                leaf_v = 1
            else:
                leaf_v = -1
            node = MctsNode(None, leaf_v, None, is_terminal=True)
        else:
            legal_action = env.legal_moves()
            if np.max(legal_action) == 0:
                node = MctsNode(None, 0, None, is_terminal=True)
            else:
                black_ary, white_ary = env.black_and_white_plane()
                state = [black_ary, white_ary] if env.player_turn() == Player.black else [white_ary, black_ary]
                future = await self.predict(np.array(state))  # type: Future
                await future
                leaf_p, leaf_v = future.result()
                _p = []
                action_map = []
                for i in range(len(legal_action)):
                    if legal_action[i]:
                        _p.append(leaf_p[i])
                        action_map.append(i)
                node = MctsNode(_p, leaf_v, action_map, is_terminal=False)
        self.var_nodes[key] = node
        self.now_expanding.remove(key)
        return node, True

    async def search_my_move(self, env: Connect4Env, node: MctsNode):
        """

        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :param env:
        :param node: MCTS node
        :param is_root_node:
        :return:
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            elif env.winner == env.player_turn():
                return 1
            else:
                return -1
        if node.is_terminal():
            return 0

        node_list = []
        is_root_node = True
        while True:
            while True:
                action_t, next_node = await node.select(is_root=is_root_node, config=self.play_config)
                node_list.append((node, action_t))
                is_root_node = False
                if action_t is not None:
                    _, _ = env.step(node.action_map[action_t])
                else:
                    logger.error(f"invalid tree ({env.turn}, {self.counter_key(env)})")

                if next_node is None or next_node.is_terminal():
                    break
                node = next_node

            # is leaf?
            leaf_v = 0
            if next_node is not None and next_node.is_terminal():
                leaf_v = next_node.reward()
                break
            else:
                next_node, is_expanded = await self.create_node(env)
                leaf_v = next_node.V
                with await node.sem:
                    node.children[action_t] = next_node
                if is_expanded or next_node.is_terminal():
                    break
                else:
                    if self.play_config.logging_thinking:
                        logger.debug("continue search node")
                    node = next_node
                    continue
        leaf_v = -leaf_v
        for node, action_t in node_list[::-1]:
            await node.backup(leaf_v, action_t, self.play_config)
            leaf_v = -leaf_v
        return leaf_v

    async def expand_and_evaluate(self, env):
        """expand new leaf

        update var_p, return leaf_v

        :param ChessEnv env:
        :return: leaf_v
        """
        key = self.counter_key(env)
        self.now_expanding.add(key)

        black_ary, white_ary = env.black_and_white_plane()
        state = [black_ary, white_ary] if env.player_turn() == Player.black else [white_ary, black_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.

        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            # logger.debug(f"predicting {len(item_list)} items")
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.api.predict(data)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        """

        :param z: win=1, lose=-1, draw=0
        :return:
        """
        self.moves.reverse()
        for move in self.moves:  # add this game winner result to all past moves.
            z = -z
            move += [z]

    def calc_policy(self, board):
        """calc π(a|s0)
        :return:
        """
        pc = self.play_config
        env = Connect4Env().update(board)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    @staticmethod
    def counter_key(env: Connect4Env):
        return CounterKey(env.observation, env.turn)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)

        legal_moves = env.legal_moves()

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]

        if is_root_node:  # Is it correct?? -> (1-e)p + e*Dir(0.03)
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.labels_n)

        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])
        if env.player_turn() == Player.white:
            v_ = (self.var_q[key] + u_ + 1000) * legal_moves
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q[key] + u_ + 1000) * legal_moves

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t
