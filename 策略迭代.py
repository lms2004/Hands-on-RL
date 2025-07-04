import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    """ 策略迭代算法（Policy Iteration） """

    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * (self.env.ncol * self.env.nrow)  # 初始化 V(s)=0
        self.pi = [[0.25, 0.25, 0.25, 0.25]              # 初始化策略为均匀分布：π(a|s)=0.25
                   for _ in range(self.env.ncol * self.env.nrow)]
        self.theta = theta  # 价值函数收敛阈值 ε
        self.gamma = gamma  # 折扣因子 γ ∈ [0, 1]

    def policy_evaluation(self):  # 策略评估 (Policy Evaluation)
        cnt = 1  # 记录迭代轮数（仅用于打印日志）
        
        while True: # 每次迭代 V ← V_k+1
            max_diff = 0  # 当前轮中所有状态价值变动的最大值（用于判断是否收敛）
            
            # 初始化一个新一轮的状态价值函数 V_k+1(s)，初始为0（稍后会填入新计算的值）
            new_v = [0] * (self.env.ncol * self.env.nrow)

            # 遍历所有状态 s ∈ 𝒮
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # 用于存储当前状态 s 下，所有动作的 π(a|s) * Q(s,a)
                
                # 遍历当前状态下的所有可能动作 a ∈ 𝒜（此处动作数为4）
                for a in range(4):
                    qsa = 0  # Qπ(s,a) 累加器

                    # 遍历所有 (p, s', r, done)：在状态 s 执行动作 a 后可能到达的后继状态 s'
                    for p, next_state, r, done in self.env.P[s][a]:
                        # p: 转移概率 P(s'|s,a)
                        # next_state: 下一状态 s'
                        # r: 即时奖励 r(s,a,s')
                        # done: 是否是终止状态

                        # Qπ(s,a) += P(s'|s,a) * [r + γ * V_k(s')]
                        # 如果是终止状态(done=True)，则不考虑后续状态价值
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    
                    # π(a|s) * Qπ(s,a)
                    qsa_list.append(self.pi[s][a] * qsa)
                
                # Vπ(s) = ∑_{a} π(a|s) * Qπ(s,a)，策略下的状态价值函数
                new_v[s] = sum(qsa_list)

                # 记录新旧价值函数的最大差异，用于判断是否收敛
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            # 更新当前的状态价值函数 V ← V_k+1
            self.v = new_v

            # 若所有状态的更新值都小于阈值 θ，说明已收敛，退出循环
            if max_diff < self.theta:
                break

            cnt += 1  # 否则继续迭代

        print("策略评估进行 %d 轮后完成" % cnt)


    def policy_improvement(self):  # 策略提升 (Policy Improvement)
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for p, next_state, r, done in self.env.P[s][a]:
                    # Qπ(s, a) = ∑_{s'} P(s'|s,a) * [r(s,a) + γ V(s')]
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            # π'(a|s) = 1/cntq if Q(a)=maxQ else 0 （均分最优动作）
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代主函数
        while True:
            self.policy_evaluation()  # 使用 π 评估 Vπ
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()  # 使用 Vπ 提升 π
            if old_pi == new_pi:  # 若策略不再变化 → π 已最优
                break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
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
theta = 0.001 # 值函数收敛的误差阈值
gamma = 0.9 # 折扣因子
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
