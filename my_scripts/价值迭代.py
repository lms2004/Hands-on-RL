import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)]
        # 含义: 在状态 state 执行动作 action
        # 得到下一个状态 next_state、奖励 reward、是否结束 done
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作: 上(0), 下(1), 左(2), 右(3)
        # 坐标系原点在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 如果当前位置是悬崖或终点, 则无法再移动
                    if i == self.nrow - 1 and j > 0:
                        # 吸收状态: 奖励=0, 直接结束
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 计算下一个位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1  # 默认每一步奖励 -1
                    done = False
                    # 如果下一个位置在悬崖或终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值函数 V(s)=0
        self.theta = theta  # 收敛阈值
        self.gamma = gamma  # 折扣因子
        # 最终策略 π
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 存放所有动作的Q(s,a)
                for a in range(4):
                    qsa = 0
                    # 公式: Q(s,a) = Σ_s' p(s'|s,a)[ r + γ V(s') ]
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)
                # 价值迭代公式:
                # V(s) ← max_a Q(s,a)
                new_v[s] = max(qsa_list)
                # 计算最大变化量
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            # 收敛条件: max_s |V_new(s) - V(s)| < θ
            if max_diff < self.theta: break
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):  
        # 根据最终的价值函数 V(s)，导出一个贪婪策略:
        # π(s) = argmax_a Q(s,a)
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    # Q(s,a) = Σ_s' p(s'|s,a)[ r + γ V(s') ]
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 有多少个动作达到最优
            # 平均分配概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:  # 悬崖
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
