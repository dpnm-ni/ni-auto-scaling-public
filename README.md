# ni-auto-scaling-module
NI-Auto-Scaling-Module applies auto-scaling to SFC in the OpenStack testbed.
SFC consists of multi-tier VNF instances and this module applies auto-scaling by choosing a tier and performing an scaling action to the SFC.  

## Main Responsibilities
Threshold or RL-based Auto-Scaling module
- Provide APIs to do auto-scaling by threshold or Deep Q-network (DQN)
- Provide APIs to get auto-scaling processes running in environment
- Provide API to delete an auto-scaling process running in environment

## Requirements
```
Python 3.5.2+
```

Please install pip3 and requirements by using the command as below.
```
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

This module requires pytorch to run DQN. However, pytorch has various options for installation so please take a look at <https://pytorch.org/> and install pytorch according to your environment.

## Configuration
This module runs as web server to handle HTTP messages required to create/get/delete an auto-scaling process.
To use a web UI of this module or send HTTP messages to the module, a port number can be configured (a default port number is 8004).

```
# server/__main__.py

def main():
    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'NI Auto-Scaling Service'})
    app.run(port=8004)
```

This module interacts with ni-mano to handle auto-scaling functions in OpenStack environment and needs configuration.

- ***ni_mon, ni_nfvo***: This module interacts with ni-mano to handle auto-scaling functions in OpenStack environment. To communicate with ni-mano, this module should know URLs of ni-mano.
- ***instance***: This module needs an account to access an instance created by an auto-scaling process and configuration to classify VNF instances in SFC. To this end, it uses *prefix_splitter* to distinguish VNF types and *max_number*/*min_number* to limit the number of instances in each tier. For example, if an instance has a name "test-firewall-1" and *prefix_splitter* is '-', this module extracts *firewall* and uses it as a VNF type.
- ***sla_monitoring***: This module measures response time through an SFC that is handled by an auto-scaling module to trigger a scaling action or calculate rewards. To generate traffic from a source, this module can access the source and generate traffic to a destination.
- ***image, flavor***: Because this module resizes the number of VNF instances in SFC it can use pre-defined images and flavor (Instance spec.) to create VNFs.  

These data should configured correctly to run this module as follows.

```
# config/config.yaml

ni_mon:
  host: http://<ni_mon_ip>:<ni_mon_port>    # Configure here to interact with a monitoring module
ni_nfvo:
  host: http://<ni_nfvo_ip>:<ni_nfvo_port>  # Configure here to interact with an NFVO module
instance:                                   # Information of new instance created by a scale-out action    
  id: <instance_ssh_id>                     # SSH ID of new VNF instance    
  password: <instance_ssh_pw>               # SSH ID of new VNF instance
  prefix_splitter: '-'                      # Prefix to classify VNF instance name
  max_number: 5                             # Maximum number of VNF instances allowed in each tier
  min_number: 1                             # Minimum number of VNF instances allowed in each tier
sla_monitoring:                             # To access traffic generator and create traffic
  src: 141.223.82.43                        # IP of traffic generator (instance in OpenStack)
  id: aca0c954-27e9-42d8-8240-689948137f28  # ID of traffic generator (instance in OpenStack)  
  ssh_id: <ssh_id of the traffic generator> # SSH ID of traffic generator
  ssh_pw: <ssh_pw of the traffic generator> # SSH PW of traffic generator
  num_requests: 100                         # Number of messages (This module generates HTTP messages)
image:                                      # Image IDs used by OpenStack
  firewall: <OpenStack Image ID>
  flowmonitor: <OpenStack Image ID>
  dpi: <OpenStack Image ID>
  ids: <OpenStack Image ID>
  proxy: <OpenStack Image ID>
  sla_monitor: <OpenStack Image ID>
flavor:                                     # Flavor ID used by OpenStack
  default:<OpenStack Flavor ID>
```

Before running this module, OpenStack network ID should be configured because VNF instances in OpenStack can have multiple network interfaces.
This module uses *openstack_network_id* value to identify a network interface used to create an SFC that is handled by an auto-scaling process.
Moreover, Deep Q-networks (DQN) hyper-parameters can be configured as follows (they have default values).

```
# auto_scaling.py

# OpenStack Parameters
openstack_network_id = "" # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 5000            # Maximum Buffer size
batch_size    = 32              # Batch size for mini-batch sampling
num_neurons = 128               # Number of neurons in each hidden layer
epsilon = 0.08                  # epsilon value of e-greedy algorithm
required_mem_size = 200         # Minimum number triggering sampling
print_interval = 20             # Number of iteration to print result during DQN
```

## Usage

After installation and configuration of this module, you can run this module by using the command as follows.

```
python3 -m server
```

This module provides web UI based on Swagger:

```
http://<host IP running this module>:<port number>/ui/
```

To handle an auto-scaling process in OpenStack testbed, this module processes HTTP POST/GET/DELETE messages.
You can generate these messages by using web UI or using other library creating HTTP messages.

The destination URLs are as follows.

### URL of REST APIs

- ***(HTTP POST)*** http://\<host IP running this module\>:\<port number\>/create_scaling/threshold
- ***(HTTP POST)*** http://\<host IP running this module\>:\<port number\>/create_scaling/dqn
- ***(HTTP GET)*** http://\<host IP running this module\>:\<port number\>/get_all_scaling
- ***(HTTP GET)*** http://\<host IP running this module\>:\<port number\>/get_scaling/{name}
- ***(HTTP DELETE)*** http://\<host IP running this module\>:\<port number\>/delete_scaling/{name}


### Create an auto-scaling process

Required data to create an auto-scaling process based on threshold or DQN is in *Threshold_ScalingInfo* model and *DQN_ScalingInfo* model respectively.
These models are JSON format data.
To create an auto-scaling process in OpenStack testbed, this module processes a HTTP POST message including the model data in its body.

#### The ***Threshold_ScalingInfo*** model consists of 6 data as follows.

- ***scaling_name***: a name of auto-scaling process identified by this module
- ***sfc_name***: a name of SFC to be applied an auto-scaling process
- ***duration***: a time to define the operating time of a created auto-scaling process. If you want to run infinitely, assign value 0 (unit: second)
- ***interval***: an interval time to trigger an auto-scaling action every interval (unit: second)
- ***threshold_in***: a value to trigger a scale-in action if response time measured is smaller than this value (unit: millisecond)
- ***threshold_out***: a value to trigger a scale-out action if response time measured is bigger than this value (unit: millisecond)

For example, if a request includes *Threshold_ScalingInfo* data as follows, this module identifies an SFC of which name is *test-sfc-1* and applies scaling actions to the SFC.
This process will be running infinitely because the *duration* value is 0 and make a scaling decision every 10 seconds according to the *interval* value.
Whenever making a decision, it depends on a response time measured through the SFC whether the measurement value is smaller or bigger than *threshold_in/out*.

```
    {
      "scaling_name": "test-threshold-scaling",
      "sfc_name": "test-sfc-1",
      "duration": 0,
      "interval": 10,
      "threshold_in": 20,
      "threshold_out": 40
    }
```

#### The ***DQN_ScalingInfo model*** consists of 6 data as follows.

- ***scaling_name***: a name of auto-scaling process identified by this module
- ***sfc_name***: a name of SFC to be applied an auto-scaling process
- ***has_dataset***: a boolean value to check whether there is a dataset to be used for training. If the dataset exists, this module reads the data from a dataset file and update the dataset during auto-scaling. The file name is same as the ***scaling_name***.
- ***duration***: a time to define the operating time of a created auto-scaling process. If you want to run infinitely, assign value 0 (unit: second)
- ***interval***: an interval time to trigger an auto-scaling action every interval (unit: second)
- ***slo***: a value of service level objective, which is response time (unit: millisecond)

For example, if a request includes *DQN_ScalingInfo* data as follows, this module identifies an SFC of which name is *test-sfc-2* and applies scaling actions to the SFC.
This process will be running during 1 hour because the *duration* value is 3600 and make a scaling decision every 20 seconds according to the *interval* value.
Additionally, it reads stored dataset from a file of which name is *test-dqn-scaling* for training because the *has_dataset* value is true.
Whenever making a decision, this module calculates rewards using the *slo* value and save a data to the file.

```
    {
      "scaling_name": "test-dqn-scaling",
      "sfc_name": "test-sfc-2",
      "has_dataset": true,
      "duration": 3600,
      "interval": 20,
      "slo": 45
    }
```

### Get information of auto-scaling processes

To get information of running auto-scaling processes in OpenStack testbed, this module processes a HTTP GET message.
This message does not need to have data in its body because it is HTTP GET message.
Instead, it needs to have *{name}* parameter in REST API URL when information of one of auto-scaling processes is required.
On the other hand, you do not need to insert any parameters if you want to get information of all processes.


### Delete an auto-scaling process

To stop and delete a running auto-scaling process, this module processes a HTTP DELETE message.
It needs to have *{name}* parameter in REST API UR:.
If there is an auto-scaling process of which name same as the *{name}*, this module stops and deletes the process.
