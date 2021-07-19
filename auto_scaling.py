import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from datetime import datetime, timedelta
from config import cfg
from torch_dqn import *

import numpy as np
import threading
import datetime as dt
import math
import os
import time
import subprocess
from pprint import pprint
import random

# Parameters
# OpenStack Parameters
openstack_network_id = "43bccef5-e205-46ab-b7f4-08bf4c94e1ae" # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 2500            # Maximum Buffer size
batch_size    = 16              # Batch size for mini-batch sampling
num_neurons = 128               # Number of neurons in each hidden layer
epsilon = 0.10                  # epsilon value of e-greedy algorithm
required_mem_size = 20          # Minimum number triggering sampling
print_interval = 20             # Number of iteration to print result during DQN

# Global values
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"
scaler_list = []
sfc_update_flag = True

# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api


# get_nfvo_sfc_api(): get ni_nfvo_sfc api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfc api
def get_nfvo_sfc_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfc_api


# get_nfvo_sfcr_api(): get ni_nfvo_sfcr api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfcr api
def get_nfvo_sfcr_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfcr_api


# get_nfvo_vnf_api(): get ni_nfvo_vnf api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf api
def get_nfvo_vnf_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_vnf_api


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"]["default"]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


# get_ip_from_vm(vm_id): get a data plane IP of VM instance
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instance(vm_id)

    ## Get ip address of specific network
    ports = query.ports
    network_id = openstack_network_id

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


# get_node_info(): get all node information placed in environment
# Input: null
# Output: Node information list
def get_node_info():
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute" and node_info.status == "enabled"]
    response = [ node_info for node_info in response if not (node_info.name).startswith("NI-Compute-82-9")]

    return response


# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple or list [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: VNF information list
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        vnfi_list = vnfi_list + temp[i]

    return vnfi_list


# get_sfcr_by_name(sfcr_name): get sfcr information by using sfcr_name from NFVO module
# Input: sfcr name
# Output: sfcr_info
def get_sfcr_by_name(sfcr_name):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.name == sfcr_name ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfcr_by_id(sfcr_id): get sfc information by using sfcr_id from NFVO module
# Input: sfcr_id, FYI, sfcr is a flow classifier in OpenStack
# Output: sfcr_info
def get_sfcr_by_id(sfcr_id):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.id == sfcr_id ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfc_by_name(sfc_name): get sfc information by using sfc_name from NFVO module
# Input: sfc name
# Output: sfc_info
def get_sfc_by_name(sfc_name):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()
    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.sfc_name == sfc_name ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info


# get_soruce_client(sfc_name): get source client ID by using sfc_name
# Input: sfc name
# Output: source client info
def get_source_client(sfc_name):

    sfc_info = get_sfc_by_name(sfc_name)
    sfcr_info = get_sfcr_by_id(sfc_info.sfcr_ids[0])
    source_client = get_monitoring_api().get_vnf_instance(sfcr_info.source_client)

    return source_client


# get_tier_status(vnf_info, sfc_info, source_node): get each tier status, each tier includes same type VNF instances
# Input: vnf_info, sfc_info, SFC's source_node
# Output: each tier status showing resource utilizataion (CPU, Memory, Number of disk operations), size, distribution)
def get_tier_status(vnf_info, sfc_info, source_node):

    ni_mon_api = get_monitoring_api()
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge",
                     "vda___disk_ops___read___derive",
                     "vda___disk_ops___write___derive"]

    vnf_instances_ids = sfc_info.vnf_instance_ids
    tier_status = []
    num_nodes = len(get_node_info())

    # Set time-period to get resources
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(seconds = 10)

    # Select each tier (VNF type)
    for tier_vnfs in vnf_instances_ids:
        tier_values = []
        tier_distribution = []

        # Select each instance ID in each tier
        for vnf in tier_vnfs:
            vnf_values = []

            for vnfi in vnf_info:
                if vnfi.id == vnf:
                    tier_distribution.append(vnfi.node_id)

            # Calculate resource status of each vnf
            for type in resource_type:

                query = ni_mon_api.get_measurement(vnf, type, start_time, end_time)
                value = 0
                memory_total = 0

                for response in query:
                    value = value + response.measurement_value

                num_query = len(query)
                value = 0 if num_query == 0 else value/num_query

                # Memory utilization calculation
                if type.startswith("memory"):
                    flavor_id = ni_mon_api.get_vnf_instance(vnf).flavor_id
                    memory_ram_mb = ni_mon_api.get_vnf_flavor(flavor_id).ram_mb
                    memory_total = 1000000 * memory_ram_mb

                    value = 100*(1-(value/memory_total)) if num_query != 0 else 0

                vnf_values.append(value) # CPU, Memory, Disk utilization of each VNF

            tier_values.append(vnf_values) # CPU, Memory, Disk utilization of each tier

        # Define status
        cpu, memory, disk = 0, 0, 0

        # Calculate sum of each resource utilization of each tier
        for vnf_values in tier_values:
            cpu = cpu + vnf_values[0]
            memory = memory + vnf_values[1]
            disk = disk + vnf_values[2] + vnf_values[3]

        # Calculate average resource utilization to define state
        tier_size = len(tier_values)

        resource = 0.70*cpu/tier_size + 0.30* memory/tier_size #+ 0.001*disk/tier_size
        tier_nodes = tier_distribution

        dist_tier = 0

        for node in tier_nodes:
            num_hops = get_hops_in_topology(source_node, node)
            dist_tier = dist_tier + num_hops/tier_size

        status = { "resource" : resource, "size" : tier_size, "distribution" : dist_tier, "placement": tier_nodes }

        tier_status.append(status)

    return tier_status


# get_state(tier_status): pre-processing tier_status to make tensor as an input data
# Input: tier_status
# Output: np.array (tensor for input values)
def get_state(tier_status):
        s = []
        for tier in tier_status:
            s.append(tier["resource"])
            s.append(tier["size"])
            s.append(tier["distribution"])

        return np.array(s)


# get_target_tier(tier_status, flag): get target tier to be applied auto-scaling action
# Input: tier_status, flag (flag is a value to check scaling in or out, positive number is scaling out)
# Output: tier index (if index is a negative number, no tier for scaling)
def get_target_tier(tier_status, flag, random_flag):

    tier_scores = []

    for tier in tier_status:
        resource_utilization = tier["resource"]
        size = tier["size"]
        dist_tier = tier["distribution"]

        if flag > 0: # Add action # Scale-out
            scaling_mask = 0.0 if size > 4 else 1.0
            dist_tier = 1.0 if dist_tier == 0 else dist_tier
            score = math.exp(resource_utilization)
            score = scaling_mask*(1/dist_tier)*score
        else: # Remove action # Scale-in
            scaling_mask = 0.0 if size < 2 else 1.0
            dist_tier = 1.0 if dist_tier == 0 else dist_tier
            score = math.exp(-resource_utilization)
            score = scaling_mask*dist_tier*score

        tier_scores.append(score)

    # Target tier has the highest value
    high_score = max(tier_scores)

    if high_score == 0: # No tier for scaling
        return -1

    if not random_flag:
        # Target selection
        return tier_scores.index(max(tier_scores))

    else:
        # Random selection
        list_for_random = [score for score in tier_scores if score > 0 ]
        return tier_scores.index(random.choice(list_for_random))


# get_scaling_target(tier_status, flag): get target to to be applied auto-scaling action
# Input: tier_status, source_node, flag (flag is a value to check scaling in or out, positive number is scaling out)
# Output: target (node id in case of scale-out, vnf id in case of scale-in)
def get_scaling_target(tier_status, source_node, flag, random_flag):

    node_info = get_node_info()
    tier_nodes = tier_status["placement"]
    target_dist = []

    if flag > 0: # scaling-out
        for node in node_info:
            if check_available_resource(node.id):
                hop = get_hops_in_topology(source_node, node.id)
                dist_tier = ((tier_status["distribution"] * tier_status["size"]) + hop) / (tier_status["size"] + 1)
                target_dist.append(dist_tier)

            else:
                target_dist.append(10000)

        if not random:
            # Target node has the highest value
            return node_info[target_dist.index(min(target_dist))].id
        else:
            # Random selection
            return random.choice(node_info).id

    else:
        for node in tier_nodes:
            hop = get_hops_in_topology(source_node, node)
            dist_tier = ((tier_status["distribution"] * tier_status["size"]) - hop) / (tier_status["size"] - 1)
            target_dist.append(dist_tier)

        if not random_flag:
            # Target instance has the highest value
            return target_dist.index(min(target_dist))

        else:
            # Random selection
            return random.randrange(0,len(target_dist))


# deploy_vnf(vnf_spec): deploy VNF instance in OpenStack environment
# Input: VnFSpec defined in nfvo client module
# Output: API response
def deploy_vnf(vnf_spec):

    ni_nfvo_api = get_nfvo_vnf_api()
    api_response = ni_nfvo_api.deploy_vnf(vnf_spec)

    return api_response


# destroy_vnf(id): destory VNF instance in OpenStack environment
# Inpurt: ID of VNF instance
# Output: API response
def destroy_vnf(id):

    ni_nfvo_api = get_nfvo_vnf_api()
    api_response = ni_nfvo_api.destroy_vnf(id)

    return api_response


# measure_response_time(): send http requests from a source to a destination
# Input: scaler
# Output: response time
def measure_response_time(scaler):

    cnd_path = os.path.dirname(os.path.realpath(__file__))
    command = "./test_http_e2e.sh %s %s %s %s %s"
    command = "cd " + cnd_path + "/testing-tools && " + command
    dst_ip = get_ip_from_id(scaler.get_monitor_dst_id())

    command = (command % (cfg["sla_monitoring"]["src"],
                          cfg["sla_monitoring"]["ssh_id"],
                          cfg["sla_monitoring"]["ssh_pw"],
                          cfg["sla_monitoring"]["num_requests"],
                          dst_ip))
    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"

    # Wait until web server is running
    start_time = dt.datetime.now()

    while True:
        time.sleep(10)

        response = subprocess.check_output(command, shell=True).strip().decode("utf-8")

        if response != "":
            pprint("[%s] %s" % (scaler.get_scaling_name(), response))
            return float(response)
        elif (dt.datetime.now() - start_time).seconds > 60 or scaler.get_active_flag() == False:
            scaler.set_active_flag(False)
            return -1


# lable_resource(flavor_id): check whether there are enough resource in nodes
# Input: node_id
# Output: True (enough), False (otherwise)
def check_available_resource(node_id):

    node_info = get_node_info()
    selected_node = [ node for node in node_info if node.id == node_id ][-1]
    flavor = get_monitoring_api().get_vnf_flavor(cfg["flavor"]["default"])

    if selected_node.n_cores_free >= flavor.n_cores and selected_node.ram_free_mb >= flavor.ram_mb:
        return True

    return False


# create_monitor(scaler): create instances to be source and destination to measure response time
# Input: scaler
# Output: True (success to create instances), False (otherwise)
def create_monitor(scaler):
    vnf_spec = get_nfvo_vnf_spec()
    vnf_spec.image_id = cfg["image"]["sla_monitor"]
    source_client = get_source_client(scaler.get_sfc_name())
    target_node = source_client.node_id

    # Repeat to try creating SLA monitors if fails
    for k in range(0, 5):

        # If enough resourcecs in a target node, create monitor
        if check_available_resource(target_node):
            vnf_spec.vnf_name = scaler.get_scaling_name() +  cfg["instance"]["prefix_splitter"] + "monitor-dst"
            vnf_spec.node_name = target_node
            dst_id = deploy_vnf(vnf_spec)

            # Wait 1 minute untill creating SLA monitors
            for i in range (0, 30):
                time.sleep(2)

                # Success to create SLA monitors
                if check_active_instance(dst_id):
                    scaler.set_monitor_src_id(cfg["sla_monitoring"]["id"])
                    scaler.set_monitor_dst_id(dst_id)
                    scaler.set_monitor_sfcr_id(create_monitor_classifier(scaler))

                    sfc_info = get_sfc_by_name(scaler.get_sfc_name())
                    sfc_info.sfcr_ids.append(scaler.get_monitor_sfcr_id())
                    update_sfc(sfc_info)

                    return True

        # If not enough resources in a target node, select another one randomly
        else:
            target_node = random.choice(get_node_info()).id


    # Fail to create SLA monitors
    destory_vnf(dst_id)

    return False


# delete_monitor(scaler): delete SLA monitor instances
# Input: scaler
# Output: Null
def delete_monitor(scaler):

    sfc_info = get_sfc_by_name(scaler.get_sfc_name())
    sfcr_id = scaler.get_monitor_sfcr_id()

    if sfcr_id in sfc_info.sfcr_ids:
        sfc_info.sfcr_ids.remove(sfcr_id)
        update_sfc(sfc_info)

        ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
        ni_nfvo_sfcr_api.del_sfcr(scaler.get_monitor_sfcr_id())

        destroy_vnf(scaler.get_monitor_dst_id())


# create_monitor_classifier(scaler): create flow classifier of traffic generator in the testbed
# Input: scaler
# Output: response
def create_monitor_classifier(scaler):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    name = scaler.get_scaling_name() + cfg["instance"]["prefix_splitter"] + "monitor"
    source_client = scaler.get_monitor_src_id()
    src_ip_prefix = get_ip_from_id(source_client) + "/32"
    dst_ip_prefix = get_ip_from_id(scaler.get_monitor_dst_id()) + "/32"
    sfcr_id = get_sfc_by_name(scaler.get_sfc_name()).sfcr_ids[-1]
    nf_chain = get_sfcr_by_id(sfcr_id).nf_chain

    sfcr_spec = ni_nfvo_client.SfcrSpec(name=name,
                                 src_ip_prefix=src_ip_prefix,
                                 dst_ip_prefix=dst_ip_prefix,
                                 nf_chain=nf_chain,
                                 source_client=source_client)

    api_response = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)

    return api_response


# get_sfc_prefix(sfc_info): get sfc_prefix from sfc_info
# Input: SFC Info.
# Output: sfc_prefix
def get_sfc_prefix(sfc_info):
    prefix = sfc_info.sfc_name.split(cfg["instance"]["prefix_splitter"])[0]
    prefix = prefix + cfg["instance"]["prefix_splitter"]

    return prefix


# update_sfc(sfc_info): Update SFC, main function to do auto-scaling
# Input: updated sfc_info, which includes additional instances or removed instances
# Output: Boolean
def update_sfc(sfc_info):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()
    sfc_update_spec = ni_nfvo_client.SfcUpdateSpec() # SfcUpdateSpec | Sfc Update info.

    sfc_update_spec.sfcr_ids = sfc_info.sfcr_ids
    sfc_update_spec.vnf_instance_ids = sfc_info.vnf_instance_ids

    for i in range(0, 15):
        time.sleep(2)

        global sfc_update_flag

        if sfc_update_flag:
            sfc_update_flag = False
            ni_nfvo_sfc_api.update_sfc(sfc_info.id, sfc_update_spec)
            sfc_update_flag = True

            return True

    return False


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    api = get_monitoring_api()
    status = api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


# threshold_scaling(scaler): doing auto-scaling based on threshold
# Input: scaler
# Output: none
def threshold_scaling(scaler):

    sfc_info = get_sfc_by_name(scaler.get_sfc_name())

    # Target SFC exist
    if sfc_info:

        # Initial Processing
        start_time = dt.datetime.now()
        source_client = get_source_client(scaler.get_sfc_name())

        flavors = get_all_flavors()
        prefix = get_sfc_prefix(sfc_info)
        instance_types = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
        del instance_types[0]

        scaler.set_active_flag(create_monitor(scaler))

        while scaler.get_active_flag():
            response_time = measure_response_time(scaler)

            if response_time < 0:
                break

            # Set sclaing_flag to show it is out or in or maintain
            if response_time > scaler.get_threshold_out():
                print("[%s] Scaling-out!" % (scaler.get_scaling_name()))
                scaling_flag = 1
            elif response_time < scaler.get_threshold_in():
                print("[%s] Scaling-in!" % (scaler.get_scaling_name()))
                scaling_flag = -1
            else:
                print("[%s] Maintain!" % (scaler.get_scaling_name()))
                scaling_flag = 0

            # Scale-in or out
            if scaling_flag != 0:
                sfc_info = get_sfc_by_name(scaler.get_sfc_name())
                vnf_info = get_vnf_info(prefix, instance_types)
                type_instances = get_instances_in_sfc(vnf_info, sfc_info)

                # Select Type
                type_status = get_type_status(type_instances, flavors)
                type_index = get_target_type(type_status, source_client.node_id, scaling_flag, True)

                #if tier_index > -1:
                if type_index > -1:
                    scaling_target = get_scaling_target(type_status[type_index], source_client.node_id, scaling_flag, True)
                    type_name = instance_types[type_index]
                    instance_ids_in_type = [ instance.id for instance in type_instances[type_index]]
                    num_instances = len(instance_ids_in_type)

                    # Scaling-out
                    if scaling_flag > 0:
                        # If possible to deploy new VNF instance
                        if num_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = prefix + type_name + cfg["instance"]["prefix_splitter"] + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            vnf_spec.image_id = cfg["image"][type_name]
                            vnf_spec.node_name = scaling_target
                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            limit = 30
                            for i in range(0, limit):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                    instance_ids_in_type.append(instance_id)
                                    sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                                    update_sfc(sfc_info)
                                    break
                                elif i == (limit-1):
                                    destroy_vnf(instance_id)


                    # Scaling-in
                    elif scaling_flag < 0:
                        # If possible to remove VNF instance
                        if num_instances > cfg["instance"]["min_number"]:
                            index = scaling_target #random.randrange(0, num_tier_instances)

                            instance_id = instance_ids_in_type[index]
                            instance_ids_in_type.remove(instance_id)
                            sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                            update_flag = update_sfc(sfc_info)

                            if update_flag:
                                destroy_vnf(instance_id)
                else:
                    pprint("[%s] No scaling because of no target tier!" % (scaler.get_scaling_name()))

            current_time = dt.datetime.now()

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            time.sleep(scaler.get_interval())


    # Delete AutoScaler object
    if scaler in scaler_list:
        delete_monitor(scaler)
        scaler_list.remove(scaler)
        pprint("[Expire: %s] Threshold Scaling" % (scaler.get_scaling_name()))
    else:
        pprint("[Exit: %s] Thresold Scaling" % (scaler.get_scaling_name()))


# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler):

    sfc_info = get_sfc_by_name(scaler.get_sfc_name())

    # Target SFC exist
    if sfc_info:

        # Initial Processing
        start_time = dt.datetime.now()
        source_client = get_source_client(scaler.get_sfc_name())
        epsilon_value = epsilon

        flavors = get_all_flavors()
        prefix = get_sfc_prefix(sfc_info)
        instance_types = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
        del instance_types[0] # Flow classifier instance deletion

        #node_info = get_node_info()

        # Q-networks
        num_states = 5 # Number of states
        num_actions = 3 # Add, Maintain, Remove

        q = Qnet(num_states, num_actions, num_neurons)
        q_target = Qnet(num_states, num_actions, num_neurons)
        q_target.load_state_dict(q.state_dict())

        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        n_epi = 0

        # If there is dataset, read it
        memory = ReplayBuffer(buffer_limit)

        # Start scaling
        scaler.set_active_flag(create_monitor(scaler))

        while scaler.get_active_flag():
            epsilon_value = 0.1

            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(prefix, instance_types)

            service_info = get_service_info(vnf_info, sfc_info, flavors)

            # Get state and select action
            s = state_pre_processor(service_info)

            decision = q.sample_action(torch.from_numpy(s).float(), epsilon_value)

            a = decision["action"]
            decision_type = "Policy" if decision["type"] else "Random"

            done = False

            # Check whether it is out or in or maintain
            if a == 0:
                print("[%s] Scaling-out! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = 1
            elif a == 2:
                print("[%s] Scaling-in! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = -1
            else:
                print("[%s] Maintain! by %s" % (scaler.get_scaling_name(), decision_type))
                scaling_flag = 0

            # Scaling in or out
            if scaling_flag != 0:

                # Scaling할 Type 선택
                type_instances = get_instances_in_sfc(vnf_info, sfc_info)
                type_status = get_type_status(type_instances, flavors)
                type_index = get_target_type(type_status, source_client.node_id, scaling_flag, False)

                if type_index > -1:
                    scaling_target = get_scaling_target(type_status[type_index], source_client.node_id, scaling_flag, False)
                    type_name = instance_types[type_index]
                    instance_ids_in_type = [ instance.id for instance in type_instances[type_index]]
                    num_instances = len(instance_ids_in_type)

                    # Scaling-out
                    if scaling_flag > 0:
                        # If possible to deploy new VNF instance
                        if num_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = prefix + type_name + cfg["instance"]["prefix_splitter"] + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            vnf_spec.image_id = cfg["image"][type_name]
                            vnf_spec.node_name = scaling_target
                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            limit = 30
                            for i in range(0, limit):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                    #tier_vnf_ids.append(instance_id)
                                    instance_ids_in_type.append(instance_id)
                                    sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                                    update_sfc(sfc_info)
                                    done = True
                                    break
                                elif i == (limit-1):
                                    destroy_vnf(instance_id)

                    # Scaling-in
                    elif scaling_flag < 0:
                        # If possible to remove VNF instance
                        if num_instances > cfg["instance"]["min_number"]:
                            index = scaling_target

                            instance_id = instance_ids_in_type[index]
                            instance_ids_in_type.remove(instance_id)
                            sfc_info.vnf_instance_ids[type_index] = instance_ids_in_type
                            update_flag = update_sfc(sfc_info)

                            if update_flag:
                                destroy_vnf(instance_id)
                                done = True

                    # Maintain
                    else:
                        done = True
                else:
                    pprint("[%s] No scaling because of no target tier!" % (scaler.get_scaling_name()))

            # Prepare calculating rewards
            time.sleep(5)
            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(prefix, instance_types)

            service_info = get_service_info(vnf_info, sfc_info, flavors)
            s_prime = state_pre_processor(service_info)

            response_time = measure_response_time(scaler)

            if response_time < 0:
                break

            type_instances = get_instances_in_sfc(vnf_info, sfc_info)
            type_status = get_type_status(type_instances, flavors)

            r = reward_calculator(service_info, response_time)

            done_mask = 1.0 if done else 0.0
            transition = (s,a,r,s_prime,done_mask)
            memory.put(transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, gamma, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                print("[%s] Target network updated!" % (scaler.get_scaling_name()))
                q_target.load_state_dict(q.state_dict())

            current_time = dt.datetime.now()

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            n_epi = n_epi+1

            if n_epi > 1000:
                scaler.set_active_flag(False)

            time.sleep(scaler.get_interval())

    # Delete AutoScaler object
    if scaler in scaler_list:
        delete_monitor(scaler)
        scaler_list.remove(scaler)
        pprint("[Expire: %s] DQN Scaling" % (scaler.get_scaling_name()))
    else:
        pprint("[Exit: %s] DQN Scaling" % (scaler.get_scaling_name()))

    q.save_model("./"+scaler.get_scaling_name())

# calculate_reward(vnf_info, sfc_info, tier_status, slo, response_time): calcuate reward about action
# Input: vnf_info, node_info, sfc_info, tier_status, slo, response_time (get data to calculate reward)
# Output: calculated reward
def calculate_reward(vnf_info, sfc_info, tier_status, slo, response_time):
    alpha = 1.0 # weight1
    beta =  1.2 # weight2
    sla_score = 0 # Check sla violation
    dist_sfc = 0 # Distribution of SFC

    # Preprocessing: get vnf IDs placed in sfc
    sfc_vnf_ids = []
    tier_size = len(sfc_info.vnf_instance_ids)

    for vnf_ids in sfc_info.vnf_instance_ids:
        sfc_vnf_ids = sfc_vnf_ids + vnf_ids

    # SLA violation check for reward
    # VNF Usage (Total VNF / tierSize) and Dist_SFC
    sla_score = -response_time/slo
    vnf_usage = len(sfc_vnf_ids)/tier_size

    for status in tier_status:
        dist_sfc = dist_sfc + status["distribution"]/tier_size

    # Calculation reward
    reward = sla_score + (alpha * (1/vnf_usage) * math.exp(-beta*dist_sfc))
    return reward

def get_all_flavors():

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_flavors()

    return query

def get_instances_in_sfc(vnf_info, sfc_info):
    instance_list = []

    for type_ids in sfc_info.vnf_instance_ids:
        type_instances = [ instance for instance in vnf_info if instance.id in type_ids ]
        instance_list.append(type_instances)

    return instance_list

def get_instance_info(instance, flavor):
    ni_mon_api = get_monitoring_api()
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge",
                     "vda___disk_ops___read___derive",
                     "vda___disk_ops___write___derive",
                     "___if_dropped___tx___derive",
                     "___if_dropped___rx___derive",
                     "___if_packets___tx___derive",
                     "___if_packets___rx___derive"]

    info = { "id": instance.id, "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "loss": 0.0, "location": "NULL" }

    # Get port names
    for port in instance.ports:
        if port.network_id == openstack_network_id:
            network_port = port.port_name
            break

    for resource in resource_type:
        if "drop" in resource or "packets" in resource:
            resource_type[resource_type.index(resource)] = network_port + resource

    # Set time-period to get resources
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(seconds = 10)

    for resource in resource_type:
        query = ni_mon_api.get_measurement(instance.id, resource, start_time, end_time)
        value = 0

        for response in query:
            value = value + response.measurement_value

        value = value/len(query) if len(query) > 0 else 0

        if resource.startswith("cpu"):
            info["cpu"] = value
        elif resource.startswith("memory"):
            memory_ram_mb = flavor.ram_mb
            memory_total = 1000000 * memory_ram_mb
            info["memory"] = 100*(1-(value/memory_total)) if len(query) > 0 else 0
        elif resource.startswith("vda"):
            info["disk"] = info["disk"] + (value/(2*1000000))
        elif "___if_dropped" in resource:
            info["drops"] = info["drops"] + value
        elif "___if_packets" in resource:
            info["packets"] = info["packets"] + value

    info["location"] = instance.node_id
    info["loss"] = info["drops"]/info["packets"] if info["packets"] > 0 else 0

    return info

def get_type_status(type_instances, flavors):

    type_status = []

    # Set time-period to get resources
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(seconds = 10)

    for type in type_instances:
        type_info = { "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "loss": 0.0, "location": [], "size": 0, "allocation": {"core": 0, "memory": 0} }
        type_size = len(type)

        type_info["size"] = type_size

        for instance in type:
            for flavor_info in flavors:
                if flavor_info.id == instance.flavor_id:
                    flavor = flavor_info
                    type_info["allocation"]["core"] = flavor_info.n_cores
                    type_info["allocation"]["memory"] = flavor_info.ram_mb
                    break

            instance_info = get_instance_info(instance, flavor)
            type_info["cpu"] = type_info["cpu"] + instance_info["cpu"]/type_size
            type_info["memory"] = type_info["memory"] + instance_info["memory"]/type_size
            type_info["disk"] = type_info["disk"] + instance_info["disk"]/type_size
            type_info["packets"] = type_info["packets"] + instance_info["packets"]/type_size
            type_info["drops"] = type_info["drops"] + instance_info["drops"]/type_size
            type_info["location"].append(instance_info["location"])

        type_info["loss"] = type_info["drops"]/type_info["packets"] if type_info["packets"] > 0 else 0

        type_status.append(type_info)

    return type_status

def get_hops_in_topology(src_node, dst_node):

    nodes = [ "NI-Compute-82-81", "NI-Compute-82-82", "NI-Compute-82-48", "NI-Compute-82-49", "NI-Compute-82-51", "NI-Compute-82-55"]
    hops = [[1, 2, 4, 4, 4, 5],
            [2, 1, 4, 4, 4, 5],
            [4, 4, 1, 2, 2, 5],
            [4, 4, 2, 1, 2, 5],
            [4, 4, 2, 2, 1, 5],
            [5, 5, 5, 5, 5, 1]]

    return hops[nodes.index(src_node)][nodes.index(dst_node)]


def state_pre_processor(service_info):

    state = []

    # Create state
    state.append(service_info["cpu"])
    state.append(service_info["memory"])
    state.append(service_info["disk"])
    state.append(service_info["drops"]/service_info["packets"])
    state.append(service_info["placement"])

    return np.array(state)

def get_service_info(vnf_info, sfc_info, flavors):
    type_instances = get_instances_in_sfc(vnf_info, sfc_info)
    type_status = get_type_status(type_instances, flavors)
    service_info = { "cpu" : 0.0, "memory": 0.0, "disk": 0.0, "packets": 0.0, "drops": 0.0, "location": [], "placement": 0.0, "num_types": 0, "size": 0 }
    size = len(type_status)

    for status in type_status:
        service_info["cpu"] = service_info["cpu"] + status["cpu"]/size
        service_info["memory"] = service_info["memory"] + status["memory"]/size
        service_info["disk"] = service_info["disk"] + status["disk"]/size
        service_info["packets"] = service_info["packets"] + status["packets"]/size
        service_info["drops"] = service_info["drops"] + status["drops"]/size
        service_info["location"] = service_info["location"] + status["location"]
        service_info["num_types"] = service_info["num_types"] + 1

    # Placement
    placement_value = 0
    source_place = get_source_client(sfc_info.sfc_name).node_id
    size = len(service_info["location"])

    for location in service_info["location"]:
        placement_value = placement_value + (get_hops_in_topology(source_place, location)/size)

    service_info["size"] = size
    service_info["placement"] = placement_value

    return service_info

def get_target_type(type_status, source_client, flag, random_flag): # decision model

    type_scores = []
    alpha = 0.85
    beta = 0.15

    for type in type_status:
        resource_utilization = (alpha*(type["cpu"]/type["allocation"]["core"])) + (beta*(type["memory"]/type["allocation"]["memory"]))
        dist = 0

        for location in  type["location"]:
            dist = dist + (get_hops_in_topology(source_client, location)/type["size"])

        if flag > 0: # Add action
            scaling_mask = 0.0 if type["size"] > 4 else 1.0
            dist = 1.0 if dist == 0 else dist
            score = (1/dist)*math.exp(resource_utilization)
            score = scaling_mask*score
        else: # Remove action
            scaling_mask = 0.0 if type["size"] < 2 else 1.0
            dist = 1.0 if dist == 0 else dist
            score = dist*math.exp(-resource_utilization)
            score = scaling_mask*score

        type_scores.append(score)

    # Target tier has the highest value
    high_score = max(type_scores)

    if high_score == 0: # No tier for scaling
        return -1

    if not random_flag:
        # Target selection
        return type_scores.index(max(type_scores))

    else:
        # Random selection
        list_for_random = [score for score in type_scores if score > 0 ]
        return type_scores.index(random.choice(list_for_random))

def reward_calculator(service_info, response_time):
    alpha = 1.0 # weight1
    beta =  1.0 # weight2
    gamma = 1.5 # weight3

    response_time = response_time/1000.0
    loss = service_info["drops"]/service_info["packets"] if service_info["packets"] != 0 else 1
    inst_count = service_info["size"]/(service_info["num_types"]*5)

    reward = -((alpha*math.log(1+response_time)+(beta*math.log(1+loss))+(gamma*math.log(1+inst_count))))

    return reward

def get_scaling_target(status, source_node, flag, random_flag): # decision model

    node_info = get_node_info()
    type_nodes = status["location"]
    target_dist = []

    # Total Dist
    total_dist = 0
    for location in status["location"]:
        total_dist = total_dist + get_hops_in_topology(source_node, location)

    # Scale-out
    if flag > 0: # scaling-out
        for node in node_info:
            if check_available_resource(node.id):
                dist = (total_dist + get_hops_in_topology(source_node, node.id))/(status["size"]+1)
                target_dist.append(dist)
            else:
                target_dist.append(10000)

        if not random_flag:
            # Target node has the highest value
            return node_info[target_dist.index(min(target_dist))].id
        else:
            # Random selection
            return random.choice(node_info).id

    # Scale-in
    else:
        for node in type_nodes:
            dist = (total_dist - get_hops_in_topology(source_node, node))/(status["size"]-1)
            target_dist.append(dist)

        if not random_flag:
            # Target instance has the highest value
            return target_dist.index(min(target_dist))

        else:
            # Random selection
            return random.randrange(0,len(target_dist))
