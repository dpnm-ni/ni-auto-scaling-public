from torch_dqn import *
from simpleInterface import *
import datetime as dt
import time, subprocess, math
from pprint import pprint

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
discount         = 0.98            # Discount factor
buffer_limit  = 2500            # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 128               # Number of neurons in each hidden layer
epsilon = 0.95                  # epsilon value of e-greedy algorithm
required_mem_size = 20        # Minimum number triggering sampling
print_interval = 20             # Number of iteration to print result during DQN
duration = 36000                 # Scaling expire Unit: seconds
interval = 12

# Thresholds
threshold_res_in = 0.001837
threshold_res_out = 0.002500
threshold_loss_in = 10
threshold_loss_out = 20

service_name = ""
service_url = ""

# DQN State 만드는 함수
def state_pre_processor():
    feature_cpu, cpu = 0.0, get_CPU_usage(service_name)
    feature_mem, mem = 0.0, get_memory_usage(service_name)
    feature_disk, disk = 0.0, get_disk_io(service_name)
    feature_net, net = 0.0, get_packet_stats(service_name)
    feature_place = get_replica_size(service_name)/len(get_nodes())

    if cpu:
        size = len(cpu)
        for metric in cpu:
            feature_cpu += metric["cpu"]/size
    if mem:
        size = len(mem)
        for metric in mem:
            feature_mem += metric["memory"]/size
    if disk:
        size = len(disk)
        for metric in disk:
            feature_disk += metric["total"]/(1000000*size)
    if net:
        size = len(net)
        for metric in net:
            if metric["total"] > 0:
                feature_net += (float(metric["drop"])/float(metric["total"]))/float(size)

    state = list([feature_cpu, feature_mem, feature_disk, feature_net, feature_place])

    return np.array(state)

# JVB Test용
def state_pre_processor_jvb():
    feature_cpu, cpu = 0.0, get_CPU_usage(service_name)
    feature_mem, mem = 0.0, get_memory_usage(service_name)
    feature_disk, disk = 0.0, get_disk_io(service_name)
    feature_net, net = 0.0, get_jvb_loss()
    feature_place = get_replica_size(service_name)/len(get_nodes())

    if cpu:
        size = len(cpu)
        for metric in cpu:
            feature_cpu += metric["cpu"]/size
    if mem:
        size = len(mem)
        for metric in mem:
            feature_mem += metric["memory"]/size

    if disk:
        size = len(disk)
        for metric in disk:
            feature_disk += metric["total"]/(1000000*size)

    if net:
        size = len(net)
        for metric in net:
            feature_net += float(metric["loss"])/float(size)

    state = list([feature_cpu, feature_mem, feature_disk, feature_net, feature_place])

    return np.array(state)

# Q-learning state 만드는 함수
def get_q_state():
    feature_cpu, cpu = 0.0, get_CPU_usage(service_name)
    feature_mem, mem = 0.0, get_memory_usage(service_name)

    if cpu:
        size = len(cpu)
        for metric in cpu:
            feature_cpu += metric["cpu"]/size
    if mem:
        size = len(mem)
        for metric in mem:
            feature_mem += metric["memory"]/size

    feature_cpu *= 100
    feature_mem *= 100

    if feature_mem >= 80: # Memory High
        if feature_cpu >= 80: # CPU High, Memory High
            s = 0
        elif feature_cpu <= 20: # CPU Low, Memory High
            s = 6
        else: # CPU Normal, Memory High
            s = 3
    elif feature_mem <= 20: # Memory Low
        if feature_cpu >= 80: # CPU High, Memory Low
            s = 2
        elif feature_cpu <= 20: # CPU Low, Memory Low
            s = 8
        else: # CPU Normal, Memory Low
            s = 5
    else: # Memory Normal
        if feature_cpu >= 80: # CPU High, Memory Normal
            s = 1
        elif feature_cpu <= 20: # CPU Low, Memory Normal
            s = 7
        else: # CPU Normal, Memory Normal
            s = 4

    return s

# Resposne time 측정 함수
def measure_response_time():
    command = "ab -n {num_request} -c 10 {url}".format(num_request=150, url=service_url )
    parsing_cmd = " | grep 'Time per request' | head -1 | awk '{print $4}'"
    command += parsing_cmd

    response = subprocess.check_output(command, shell=True).strip().decode("utf-8")

    if response:
        return float(response)/1000.0

    print("Response time cannot be measured! Please check the health of the service!")
    return

# Packet drop rates 측정 함수
def measure_drop_rates():
    drop_rates, net = 0.0, get_packet_stats(service_name)
    if net:
        size = len(net)
        for metric in net:
            if metric["total"] > 0:
                drop_rates += (float(metric["drop"])/float(metric["total"]))/float(size)

    return drop_rates

# JVB Test용 Packet drop rates 측정 함수
def measure_drop_rates_jvb():
    drop_rates, net = 0.0, get_jvb_loss()
    if net:
        size = len(net)
        for metric in net:
            if metric["loss"] > 0:
                drop_rates += float(metric["loss"])/float(size)

    return drop_rates

# Reward 계산 함수 (Packet loss, response time, the number of containers로 측정)
def reward_calculator():
    w1, w2, w3 = 1.0, 1.0, 1.5 # weights
    packet_loss = measure_drop_rates()
    num_containers = get_replica_size(service_name)/5.0
    response_time = measure_response_time()

    if not response_time:
        return

    reward = -((w1*math.log(1+response_time)+(w2*math.log(1+packet_loss))+(w3*math.log(1+num_containers))))

    return reward

# Reward 계산 함수 (Packet loss, the number of containers로 측정)
def reward_calculator_wo_responsetime():
    w1, w2 = 1.0, 1.5 # weights
    packet_loss = measure_drop_rates()
    num_containers = get_replica_size(service_name)/5.0

    reward = -((w1*math.log(1+packet_loss))+(w2*math.log(1+num_containers)))

    return reward

# JVB 시나리오 Reward 계산 함수 (Packet loss, the number of containers로 측정)
def reward_calculator_jvb():
    w1, w2 = 1.0, 1.5 # weights
    packet_loss = measure_drop_rates_jvb()
    num_containers = get_replica_size(service_name)/5.0

    reward = -((w1*math.log(1+packet_loss))+(w2*math.log(1+num_containers)))

    return reward

# CSV 저장 파일
def write_metrics(path, reward, response_time):
    f = open(path + ".csv", 'a', encoding='utf-8')
    wr = csv.writer(f)
    #state = state_pre_processor()
    state = state_pre_processor_jvb()
    data = [dt.datetime.now(), response_time, reward]

    for metric in state:
        data.append(metric)

    wr.writerow(data)
    f.close()

# Perform Scaling
def scaling(action):
    # Check whether it is out or in or maintain
    if action == 0:
        print("[{}] Scale-out!".format(service_name))
        num_containers = get_replica_size(service_name)

        if num_containers == 5:
            print("Cannot add a container!")
            return False
        else:
            if scale_service(service_name, num_containers+1):
                print("Successfully add a new continer (Current containers: {})".format(num_containers+1))
                return True
            else:
                print("Fail adding a new container")
                return False

    elif action == 2:
        print("[{}] Scale-in!".format(service_name))
        num_containers = get_replica_size(service_name)

        if num_containers == 1:
            print("Cannot remove a container!")
            return False
        else:
            if scale_service(service_name, num_containers-1):
                print("Successfully remove a continer (Current containers: {})".format(num_containers-1))
                return True
            else:
                print("Fail removing a container")
                return False
    else:
        print("[{}] Maintain!".format(service_name))
        return True

# Q-learning
def q_scaling():
    '''
    State = [ 0 [CPU High,   Memory High], 1 [CPU High,   Memory Normal], 2 [CPU High,   Memory Low],
              3 [CPU Normal, Memory High], 4 [CPU Normal, Memory Normal], 5 [CPU Normal, Memory Low],
              6 [CPU Low,    Memory High], 7 [CPU Low,    Memory Normal], 8 [CPU Low,    Memory Low]]

              High 기준: 80~100
              Normal 기준:  20~80
              Low 기준: 0~20
    '''

    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        #global epsilon
        start_time = dt.datetime.now()
        n_epi = 0
        Q_table = np.zeros((9, 3))
        #Q_table = np.loadtxt("model/q_learning")
        epsilon = 0.95

        while True:
            n_epi += 1
            state, coin = get_q_state(), random.random()
            epsilon = max(0.10, epsilon*0.99)
            action = random.randrange(0,3) if coin < epsilon else np.argmax(Q_table[state,:])

            scaling(action)

            time.sleep(3)

            new_state = get_q_state()

            # 보상계산 후, Q-table 갱신
            #reward = reward_calculator()
            reward = reward_calculator_jvb()

            if not reward:
                return

            #reward = reward_calculator_wo_responsetime()
            Q_table[state, action] = Q_table[state, action] + (learning_rate*(reward+(discount*np.nanmax(Q_table[new_state, :])-  Q_table[state, action])))
            write_metrics("data/{}_{}".format(service_name, "q_learning"), reward, response_time)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 1000:
                print("[{}] Scaling process exits!".format(service_name))
                return

            time.sleep(interval)

# Q-learning
def q_training_scaling():
    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        epsilon = 0.99
        start_time = dt.datetime.now()
        n_epi = 0
        Q_table = np.zeros((9, 3))

        while True:
            n_epi += 1
            state, coin = get_q_state(), random.random()
            epsilon = max(0.01, epsilon*0.99)
            action = random.randrange(0,3) if coin < epsilon else np.argmax(Q_table[state,:])

            scaling(action)

            time.sleep(3)

            new_state = get_q_state()

            # 보상계산 후, Q-table 갱신
            #reward = reward_calculator()
            reward = reward_calculator_jvb()

            if not reward:
                return

            #reward = reward_calculator_wo_responsetime()
            Q_table[state, action] = Q_table[state, action] + (learning_rate*(reward+(discount*np.nanmax(Q_table[new_state, :]) - Q_table[state, action])))
            write_metrics("data/{}_{}".format(service_name, "q_learning_training"), reward, response_time)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 1000:
                print("[{}] Scaling process exits!".format(service_name))
                np.savetxt("model/q_learning", Q_table)

                return

            time.sleep(interval)

# Response time Threshold based scaling
def response_threshold_scaling():
    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        start_time = dt.datetime.now()
        n_epi = 0

        while True:
            n_epi += 1
            response_time = measure_response_time()

            # 응답시간 측정 문제 발생
            if not response_time:
                return

            if response_time > threshold_res_out:
                action = 0
            elif response_time < threshold_res_in:
                action = 2
            else:
                action = 1

            scaling(action)

            time.sleep(3)

            reward = reward_calculator()

            if not reward:
                return

            write_metrics("data/{}_{}".format(service_name, "response_threshold"), reward, response_time)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 1000:
                print("[{}] Scaling process exits!".format(service_name))
                return

            time.sleep(interval)

# Packet loss based threshold scaling
def drop_threshold_scaling():
    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        start_time = dt.datetime.now()
        n_epi = 0

        while True:
            n_epi += 1
            #drop_rates = measure_drop_rates()
            drop_rates = measure_drop_rates_jvb()

            if drop_rates > threshold_loss_out:
                action = 0
            elif drop_rates < threshold_loss_in:
                action = 2
            else:
                action = 1

            scaling(action)

            time.sleep(3)

            #reward = reward_calculator_wo_responsetime()
            reward = reward_calculator_jvb()

            if not reward:
                return

            write_metrics("data/{}_{}".format(service_name, "drop_threshold"), reward, None)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 1000:
                print("[{}] Scaling process exits!".format(service_name))
                return

            time.sleep(interval)

# Random scaling
def random_scaling():
    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        start_time = dt.datetime.now()
        n_epi = 0

        while True:
            action = random.randrange(0,3)

            # Check whether it is out or in or maintain
            scaling(action)

            time.sleep(3)

            #response_time = measure_response_time()
            drop_rates = get_jvb_loss() # JVB의 Loss rate 측정
            write_metrics("data/{}_{}".format(service_name, "random"), None, drop_rates) #response_time)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 2000:
                print("[{}] Scaling process exits!".format(service_name))
                return

            time.sleep(interval)

# DQN version
def dqn_scaling():
    # 서비스가 존재하면 시작
    if get_service_id(service_name):
        #global epsilon
        epsilon = 0.95
        start_time = dt.datetime.now()

        num_states, num_actions = 5, 3 # DQN 모델의 State, Action 개수
        q, q_target, memory = Qnet(num_states, num_actions, num_neurons),  Qnet(num_states, num_actions, num_neurons), ReplayBuffer(buffer_limit)
        #q.load_state_dict(torch.load("model/dqn_model")) # 학습된 모델 읽어오게 하기
        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        q_target.load_state_dict(q.state_dict())
        n_epi = 0

        # Auto-scaling 시작
        while True:
            # 모니터링 데이터 주기적으로 받기
            state = state_pre_processor_jvb() #state_pre_processor()
            epsilon = max(0.10, epsilon*0.99)
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)["action"]
            done_mask = 1.0 if scaling(action) else 0.0
            n_epi += 1

            time.sleep(3)

            new_state = state_pre_processor_jvb() #state_pre_processor()

            # 보상계산 후, Q-table 갱신
            reward = reward_calculator_jvb() #reward_calculator()

            if not reward:
                return

            # Replay memory 업데이트 과정
            transition = (state, action, reward, new_state, done_mask)
            memory.put(transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, discount, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                print("[{}] Target network updated!".format(service_name))
                q_target.load_state_dict(q.state_dict())

            write_metrics("data/{}_{}".format(service_name, "dqn"), reward, response_time)

            if (dt.datetime.now()-start_time).seconds > duration or n_epi > 1000:
                print("[{}] Scaling process exits!".format(service_name))
                torch.save(q.state_dict(), "model/dqn_model_jvb")
                return

            time.sleep(interval)
