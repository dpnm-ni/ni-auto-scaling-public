from influxdb_client import InfluxDBClient
from influxdb import InfluxDBClient as InfluxDBClient_old
import docker
from pprint import pprint

# InFluxDB Configuration
db_url = ""
token = ""
org = ""
database = ""
interval = ""

db_url_jitsi = ""

# Docker Engine Configuration
docker_url = ""

# Data source: Docker Swarm
def get_service_id(service_name):
    client = docker.DockerClient(base_url=docker_url)
    services = client.services.list()

    for service in services:
        if service.attrs["Spec"]["Name"] == service_name:
            return service.attrs["ID"]

def get_replica_size(service_name):
    client = docker.DockerClient(base_url=docker_url)
    service_id = get_service_id(service_name)

    if service_id:
        service = client.services.get(service_id).attrs
        return service["Spec"]["Mode"]["Replicated"]["Replicas"]

def get_nodes():
    client = docker.DockerClient(base_url=docker_url)
    nodes = client.nodes.list()
    result = list()

    for node in nodes:
        if node.attrs["Status"]["State"] == "ready":
            result.append({"id": node.attrs["ID"],
                           "name": node.attrs["Description"]["Hostname"],
                           "ip": node.attrs["Status"]["Addr"]})
    return result

def get_tasks(service_name):
    client = docker.DockerClient(base_url=docker_url)
    service_id = get_service_id(service_name)

    if service_id:
        tasks = client.services.get(service_id).tasks()
        tasks = [ task for task in tasks if task["Status"]["State"] == "running"]
        return tasks

def get_service_info(service_name):
    client = docker.DockerClient(base_url=docker_url)
    service_id = get_service_id(service_name)

    if service_id:
        service = client.services.get(service_id).attrs
        return service
    return

def scale_service(service_name, replica):
    client = docker.DockerClient(base_url=docker_url)
    service_id = get_service_id(service_name)

    if service_id:
        service = client.services.get(service_id)
        return service.scale(replica)


# Data source: InfluxDB
# Get CPU usage (0.0~1.0) per container
def get_CPU_usage(service_name):
    query = 'from(bucket: "%s") \
      |> range(start: -%s) \
      |> filter(fn: (r) => r["com.docker.swarm.service.name"] == "%s") \
      |> filter(fn: (r) => r["_measurement"] == "docker_container_cpu") \
      |> filter(fn: (r) => r["container_status"] == "running") \
      |> filter(fn: (r) => r["_field"] == "usage_percent") \
      |> mean() \
      |> map(fn: (r) => ({ container_name: r.container_name, _value: r._value/100.0}))' % (database, interval, service_name)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)
    results = list()

    for table in response:
        for row in table.records:
            results.append({"container_name": row["container_name"], "cpu": row["_value"]})

    return results

# Get Memory usage (0.0~1.0) per container
def get_memory_usage(service_name):
    query = 'mem = from(bucket: "%s") \
                |> range(start: -%s) \
                |> filter(fn: (r) => r["com.docker.swarm.service.name"] == "%s" and r["_measurement"] == "docker_container_mem" and r["container_status"] == "running") \
                |> filter(fn: (r) => r["_field"] == "limit" or r["_field"] == "usage") \
                |> mean() \
             tmp1 = mem |> filter(fn: (r) => r["_field"] == "limit") \
             tmp2 = mem |> filter(fn: (r) => r["_field"] == "usage") \
             join(tables:{tmp1, tmp2}, on: ["container_name"]) \
                |> map(fn:(r) => ({ container_name: r.container_name, _value: r._value_tmp2/r._value_tmp1}))' % (database, interval, service_name)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)
    results = list()

    for table in response:
        for row in table.records:
            results.append({"container_name": row["container_name"], "memory": row["_value"]})

    return results

# Get Disk operations (page in/out)
def get_disk_io(service_name):
    query = 'pages = from(bucket: "%s") \
                      |> range(start: -%s) \
                      |> filter(fn: (r) => r["com.docker.swarm.service.name"] == "%s" and r["_measurement"] == "docker_container_mem") \
                      |> filter(fn: (r) => r["_field"] == "total_pgpgin" or r["_field"] == "total_pgpgout") \
                      |> spread() \
             pgin = pages |> filter(fn: (r) => r["_field"] == "total_pgpgin") \
             pgout = pages |> filter(fn: (r) => r["_field"] == "total_pgpgout") \
             join(tables:{pgin, pgout}, on: ["container_name"]) \
                |> map(fn: (r) => ({container_name: r.container_name, total: r._value_pgin + r._value_pgout, page_in: r._value_pgin, page_out: r._value_pgout }))' % (database, interval, service_name)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)
    results = list()

    for table in response:
        for row in table.records:
            results.append({"container_name": row["container_name"], "total": row["total"], "page_in": row["page_in"], "page_out": row["page_out"]})

    return results

# Get Packet stats per container
def get_packet_stats(service_name):
    query = 'pkts = from(bucket: "%s") |> range(start: -%s) \
                        |> filter(fn: (r) => r["com.docker.swarm.service.name"] == "%s" and r["_measurement"] == "docker_container_net" and r["network"] == "total") \
                        |> filter(fn: (r) => r["_field"] == "tx_packets" or r["_field"] == "rx_packets" or r["_field"] == "tx_dropped" or r["_field"] == "rx_dropped") \
                        |> spread() \
             total_tx = pkts |> filter(fn: (r) => r["_field"] == "tx_packets") \
             total_rx = pkts |> filter(fn: (r) => r["_field"] == "rx_packets") \
             drop_tx = pkts |> filter(fn: (r) => r["_field"] == "tx_dropped") \
             drop_rx = pkts |> filter(fn: (r) => r["_field"] == "rx_dropped") \
             total = join(tables: {total_tx, total_rx}, on: ["container_name"]) \
             drop = join(tables: {drop_tx, drop_rx}, on: ["container_name"]) \
             join(tables: {total, drop}, on: ["container_name"]) \
              |> map(fn: (r) => ({ container_name: r.container_name,  \
                                   drop: r._value_drop_tx + r._value_drop_rx, drop_tx: r._value_drop_tx, drop_rx: r._value_drop_rx, \
                                   total: r._value_total_tx + r._value_total_rx, total_tx: r._value_total_tx, total_rx: r._value_total_rx })) ' % (database, interval, service_name)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)
    results = list()

    for table in response:
        for row in table.records:
            results.append({"container_name": row["container_name"],
                            "total": row["total"], "total_tx": row["total_tx"], "total_rx": row["total_rx"],
                            "drop": row["drop"], "drop_tx": row["drop_tx"], "drop_rx": row["drop_rx"]})

    return results

def get_running_container(service_name):
    query = 'from(bucket: "%s") \
              |> range(start: -%s) \
              |> filter(fn: (r) => r["com.docker.swarm.service.name"] == "%s") \
              |> filter(fn: (r) => r["_measurement"] == "docker_container_status") \
              |> filter(fn: (r) => r["container_status"] == "running") \
              |> last() \
              |> pivot(rowKey:["container_name"], columnKey: ["_field"], valueColumn: "_value") \
              |> map(fn: (r) => ({name: r.container_name, id: r.container_id, placement: r.engine_host}))' % (database, interval, service_name)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)
    result = list()

    for table in response:
        for row in table.records:
            result.append(row.values)

    return result

def get_node_spec():
    query = 'from(bucket: "%s") \
      |> range(start: -%s) \
      |> filter(fn: (r) => r["_measurement"] == "docker") \
      |> filter(fn: (r) => r["_field"] == "n_cpus" or r["_field"] == "memory_total") \
      |> pivot(rowKey:["engine_host"], columnKey: ["_field"], valueColumn: "_value") \
      |> map(fn: (r) => ({name: r.engine_host, id: r.host, cpu: r.n_cpus, memory: r.memory_total}))' % (database, interval)

    query_api = InfluxDBClient(url=db_url, token=token, org=org).query_api()
    response = query_api.query(query)

    result = list()
    for table in response:
        for row in table.records:
            result.append(row.values)

    return result

def get_jvb_loss():
    nodes = get_nodes()
    query_api = InfluxDBClient_old(host=db_url_jitsi, database=database)

    result = list()
    for node in nodes:
        query = 'SELECT last("incoming_loss"), last("outgoing_loss") FROM "jitsi_stats" WHERE ("host" =~ /^%s$/) AND time >= now() - %s GROUP BY time(5s)' % (node["name"], interval)

        response = query_api.query(query=query, epoch="ms")

        if response:
            item, size = {"host": node["name"], "loss": 0.0}, len(list(response.get_points()))
            for point in response.get_points():
                point["last"] = 0.0 if point["last"] == None else point["last"]
                point["last_1"] = 0.0 if point["last_1"] == None else point["last_1"]

                item["loss"] += (point["last"]+point["last_1"])/size

            result.append(item)

    return result
