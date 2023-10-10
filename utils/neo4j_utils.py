from __future__ import print_function, unicode_literals
from neo4j import GraphDatabase
from pandas import DataFrame
import networkx as nx
import os, glob
import pickle as pkl
from multiprocessing import Process, Queue, set_start_method
import time
import statistics


class Driver():
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", pwd="neo4j"):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd),
                                                 max_connection_lifetime=200)
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


def nx2create(gidx: str, nx_g: nx.MultiDiGraph):
    nqueries = []
    for n, ndata in nx_g.nodes(data=True):
        nname = f"n{gidx}_{n}"
        ndata_str = "{" + ', '.join([f'{k}:"{v}"' if isinstance(v, str)
                                     else f'{k}:{v}' for k, v in ndata.items()
                                     ] + [f'name:"{nname}"']) + "}"
        nqueries.append(f'({nname}:node {ndata_str})')
    nquery = ', '.join(nqueries)

    equeries = []
    for i, (u, v, k, edata) in enumerate(
                nx_g.edges(keys=True, data=True)):
        ename = f"r{i}"
        etype = edata.get('etype', 0)
        edata_str = '{' + ', '.join(
            [f'{k}:"{v}"'
             if isinstance(v, str) else f'{k}:{v}'
             for k, v in edata.items() if k != 'etype'] +
            [f'key:{k}, name:"r{i}"']) + "}"
        equeries.append(
            f'(n{gidx}_{u})-[{ename}:et{etype} {edata_str}]->(n{gidx}_{v})')

    equery = ', '.join(equeries)
    return f"CREATE {nquery}, {equery}"


def nx2query(nx_g: nx.MultiDiGraph, start=0, limit=None):
    nqueries = []
    nnames = [f"n{n}" for n in nx_g.nodes()]
    for n, ndata in nx_g.nodes(data=True):
        nname = f"n{n}"
        nnames.append(nname)
        ndata_str = "{" + ', '.join([f'{k}:"{v}"' if isinstance(v, str)
                                     else f'{k}:{v}' for k, v in ndata.items()
                                     ]) + "}"
        nqueries.append(f'({nname}:node {ndata_str})')

    nquery = ', '.join(nqueries)
    mutually_exclusive_queries = []
    for i, (n, ndata) in enumerate(nx_g.nodes(data=True)):
        nname = f"n{n}"
        not_in = ' AND '.join([f'{nname} <> {iname}' for iname in nnames[i:]
                               if iname != nname])
        if not_in:
            mutually_exclusive_queries.append(not_in)
    if mutually_exclusive_queries:
        meq = ' AND '.join(mutually_exclusive_queries)
    else:
        meq = ''

    equeries = []
    es = []
    for i, (u, v, k, edata) in enumerate(
                nx_g.edges(keys=True, data=True)):
        ename = f"r{i}"
        etype = edata.get('etype', 0)
        edata_str = '{' + ', '.join(
            [f'{k}:"{v}"'
             if isinstance(v, str) else f'{k}:{v}'
             for k, v in edata.items() if k != 'etype']) + "}"
        equeries.append(
            f'(n{u})-[{ename}:et{etype} {edata_str}]->(n{v})')
        es.append(ename)

    equery = ', '.join(equeries)
    rquery = ', '.join([f"n{n}" for n in nx_g.nodes()] +es )
    where_clause = '' if meq == '' else f'WHERE {meq} '
    return f"MATCH {nquery}, {equery} {where_clause} RETURN {rquery} SKIP {start} " +\
        (f'LIMIT {limit}' if limit else '')


def nx2query_within_set(nx_g: nx.MultiDiGraph, node_set, start=0, limit=None):
    nqueries = []
    nnames = [f"n{n}" for n in nx_g.nodes()]
    for n, ndata in nx_g.nodes(data=True):
        nname = f"n{n}"
        nnames.append(nname)
        ndata_str = "{" + ', '.join([f'{k}:"{v}"' if isinstance(v, str)
                                     else f'{k}:{v}' for k, v in ndata.items()
                                     ]) + "}"
        nqueries.append(f'({nname}:node {ndata_str})')

    nquery = ', '.join(nqueries)
    mutually_exclusive_queries = []
    for i, (n, ndata) in enumerate(nx_g.nodes(data=True)):
        nname = f"n{n}"
        not_in = ' AND '.join([f'{nname} <> {iname}' for iname in nnames[i:]
                               if iname != nname])
        if not_in:
            mutually_exclusive_queries.append(not_in)
    if mutually_exclusive_queries:
        meq = ' AND '.join(mutually_exclusive_queries)
    else:
        meq = ''

    # node set filter
    node_set_filter = f'ny.name in {node_set}'
    meq = f'{meq} AND {node_set_filter}' if meq else node_set_filter

    equeries = []
    es = []
    for i, (u, v, k, edata) in enumerate(
                nx_g.edges(keys=True, data=True)):
        ename = f"r{i}"
        etype = edata.get('etype', 0)
        edata_str = '{' + ', '.join(
            [f'{k}:"{v}"'
             if isinstance(v, str) else f'{k}:{v}'
             for k, v in edata.items() if k != 'etype']) + "}"
        equeries.append(
            f'(n{u})-[{ename}:et{etype} {edata_str}]->(n{v})')
        es.append(ename)

    equery = ', '.join(equeries)
    rquery = ', '.join([f"n{n}" for n in nx_g.nodes()] +es )
    where_clause = '' if meq == '' else f'WHERE {meq} '
    return f"MATCH {nquery}, {equery} {where_clause} RETURN {rquery} SKIP {start} " +\
        (f'LIMIT {limit}' if limit else '')

def nx2query_oos(nx_g: nx.MultiDiGraph, start=0, limit=None):
    nqueries = []
    nnames = []

    for n, ndata in nx_g.nodes(data=True):
        nname = f"n{n}"
        ndata_str = "{" + ', '.join([f'{k}:"{v}"' if isinstance(v, str)
                                     else f'{k}:{v}' for k, v in ndata.items()
                                     ]) + "}"
        nqueries.append(f'({nname}:node {ndata_str})')
        nnames.append(nname)
    nquery = ', '.join(nqueries)

    mutually_exclusive_queries = []
    for i, (n, ndata) in enumerate(nx_g.nodes(data=True)):
        nname = f"n{n}"
        not_in = ' AND '.join([f'{nname} <> {iname}' for iname in nnames[i:]
                               if iname != nname])
        if not_in:
            mutually_exclusive_queries.append(not_in)
    if mutually_exclusive_queries:
        meq = ' AND '.join(mutually_exclusive_queries)
    else:
        meq = ''

    equeries = []
    es = []
    for i, (u, v, k, edata) in enumerate(
                nx_g.edges(keys=True, data=True)):
        ename = f"r{i}"
        etype = edata.get('etype', 0)
        edata_str = '{' + ', '.join(
            [f'{k}:"{v}"'
             if isinstance(v, str) else f'{k}:{v}'
             for k, v in edata.items() if k != 'etype']) + "}"
        equeries.append(
            f'(n{u})-[{ename}:et{etype} {edata_str}]->(n{v})')
        es.append(ename)
    equery = ', '.join(equeries)

    # finally, the equivalent
    equiv_queries = []
    ne_e_pair = []
    for n, ndata in nx_g.nodes(data=True):
        nname = f'ne{n}'
        ename = f'ee{n}'
        not_in = ' AND '.join([f'{nname} <> {iname}' for iname in nnames])
        equiv_queries.append(
            f'OPTIONAL MATCH ({nname}:node)-[{ename}]->(n{n}) WHERE {not_in}')
        ne_e_pair.append((nname, ename))
    equiv_query = ' '.join(equiv_queries)
    rquery = ', '.join([f"n{n}" for n in nx_g.nodes()] + es +
                       [f'collect([{pair[0]}, {pair[1]}])'
                        for pair in ne_e_pair])
    where_clause = '' if meq == '' else f'WHERE {meq} '
    return f"MATCH {nquery}, {equery} {where_clause} {equiv_query} " +\
        f"RETURN {rquery} SKIP {start} " + (f'LIMIT {limit}' if limit else '')


def match2nx(nx_q: nx.MultiDiGraph, match_result):
    print("Matched with ", len(match_result), " records")
    outputs = []
    for record in match_result:
        # with original node name and original edge in query graph as the key
        new_graph = nx.MultiDiGraph()
        for node in nx_q.nodes():
            nname = f"n{node}"
            ndata = record[nname]
            new_graph.add_node(node, **ndata)
        for i, (u, v, k, old_e) in enumerate(nx_q.edges(keys=True, data=True)):
            ename = f"r{i}"
            edata = record[ename]
            additional_data = {}
            if 'etype' not in edata:
                additional_data['etype'] = old_e.get('etype', 0)
            new_graph.add_edge(u, v, **edata, **additional_data)
        outputs.append(new_graph)
    return outputs


def match_oos2nx(nx_q: nx.MultiDiGraph, match_result):
    outputs = []
    print("Matched with ", len(match_result), " records")
    for record in match_result:
        # with original node name and original edge in query graph as the key
        new_graph = nx.MultiDiGraph()
        for node in nx_q.nodes():
            nname = f"n{node}"
            ndata = record[nname]
            new_graph.add_node(node, **ndata)
        for i, (u, v, k, old_e) in enumerate(nx_q.edges(keys=True, data=True)):
            ename = f"r{i}"
            edata = record[ename]
            additional_data = {}
            if 'etype' not in edata:
                additional_data['etype'] = old_e.get('etype', 0)
            new_graph.add_edge(u, v, **edata, **additional_data)
        oos_pair = {}
        for n in nx_q.nodes():
            nname = f'ne{n}'
            ename = f'ee{n}'
            ne_pairs = record[f'collect([{nname}, {ename}])']
            oos_pair[n] = []
            for ne_pair in ne_pairs:
                if ne_pair[0] and ne_pair[1]:
                    oos_pair[n].append((dict(ne_pair[0]), dict(ne_pair[1])))
        outputs.append((new_graph, oos_pair))
    return outputs


def csv2query_oos_single(csv_fp, col_num, start=0, stop=None):
    cwd = os.getcwd()
    if not csv_fp.startswith('/'):
        csv_fp = f"{cwd}/{csv_fp}"
    query = f"LOAD CSV WITH HEADERS FROM \"file://{csv_fp}\" as row " + \
        f"WITH row SKIP {start} " + (f"LIMIT {stop} " if stop else '') + \
        f"MATCH (ne)-[ee]->(n {{name:row.name{col_num}}}) " +\
        f"WHERE NOT ne.name in [k in KEYS(row) | row[k]] " +\
        f"return linenumber() as rid, collect([ne, ee])"
    return query


def match_oos_single2list(match_result):
    if match_result is None:
        return {}
    N = len(match_result)
    print(f"Matched {N} records")
    oos_pairs = {}
    for record in match_result:
        ne_pairs = record[f'collect([ne, ee])']
        rid = record[f'rid']
        oos_pairs[rid] = []
        for ne_pair in ne_pairs:
            if ne_pair[0] and ne_pair[1]:
                oos_pairs[rid].append((dict(ne_pair[0]), dict(ne_pair[1])))
    return oos_pairs

def query_create_nx_instances(args):
    save_dir_val = os.path.join(args.result_dir, 'nx_g_val')
    save_dir_train = os.path.join(args.result_dir, 'nx_g_train')
    print(save_dir_train, save_dir_val)
    conn = Driver(uri="bolt://localhost:7687", user="neo4j", pwd="neo4j")
    train_db = f'train{args.db}'
    val_db = f'val{args.db}'
    conn.query(f'CREATE DATABASE {train_db} IF NOT EXISTS')
    conn.query(f'CREATE DATABASE {val_db} IF NOT EXISTS')

    train_paths = glob.glob(f'{save_dir_train}/*.pkl')
    records = conn.query("match (n) WHERE n.name STARTS WITH \"nt\" RETURN DISTINCT SPLIT(n.name, \"_\")[0]",
                            db=train_db)
    if records is None:
        names = []
    else:
        names = [int(record['SPLIT(n.name, "_")[0]'][2:]) for record in records]
    names = set(names)

    deltas = []
    for fp in train_paths:
        gidx = int(os.path.split(fp)[-1].split('.')[0].split('_')[-1])
        if gidx in names:
            continue
        nx_g = pkl.load(open(fp, 'rb'))
        gidx_neo4j = f't{gidx}'
        start = time.time()
        conn.query(nx2create(gidx_neo4j, nx_g), db=train_db)
        deltas.append(time.time() - start)
    val_paths = glob.glob(f'{save_dir_val}/*.pkl')

    records = conn.query("match (n) WHERE n.name STARTS WITH \"nv\" RETURN DISTINCT SPLIT(n.name, \"_\")[0]",
                            db=val_db)
    if records is None:
        names = []
    else:
        names = [int(record['SPLIT(n.name, "_")[0]'][2:]) for record in records]
    names = set(names)

    for fp in val_paths:
        gidx = int(os.path.split(fp)[-1].split('.')[0].split('_')[-1])
        if gidx in names:
            continue

        nx_g = pkl.load(open(fp, 'rb'))
        gidx_neo4j = f'v{gidx}'
        start = time.time()
        conn.query(nx2create(gidx_neo4j, nx_g), db=val_db)
        deltas.append(time.time() - start)
    if deltas:
        m_delta, std_delta = statistics.mean(deltas), statistics.stdev(deltas)
        print(f"Took {m_delta}, std {std_delta} seconds on average for interfacing")


def csv2query_oos(queue):
    """Read from the queue; this spawns as a separate Process"""
    conn = Driver()
    while True:
        msg = queue.get()
        if msg[0] == "DONE":
            break
        # start processing
        _, csv_fp, col, itr, thres_oos, db = msg
        out_fname = os.path.splitext(os.path.split(csv_fp)[1])[0] + f"_{col}_{itr}_{thres_oos}.pkl"
        out_fp = os.path.join(os.path.split(csv_fp)[0], out_fname)
        print(out_fp, flush=True)
        if not os.path.exists(out_fp):
            query = csv2query_oos_single(csv_fp, col, itr*thres_oos, thres_oos)
            print("next query: ", query)
            match_result = conn.query(query, db=db)
            matched_oos_col = match_oos_single2list(match_result)
            pkl.dump(matched_oos_col, open(out_fp, 'wb'))


def start_csv2query_procs(qq, num_of_reader_procs):
    """Start the reader processes and return all in a list to the caller"""
    set_start_method('fork', force=True)
    all_reader_procs = list()
    for ii in range(0, num_of_reader_procs):
        ### reader_p() reads from qq as a separate process...
        ###    you can spawn as many reader_p() as you like
        ###    however, there is usually a point of diminishing returns
        reader_p = Process(target=csv2query_oos, args=((qq),))
        reader_p.daemon = True
        reader_p.start()  # Launch reader_p() as another proc

        all_reader_procs.append(reader_p)
    return all_reader_procs


if __name__ == '__main__':
    # Before this, have to run ALTER USER neo4j SET PASSWORD 'neo4j'
    conn = Driver(uri="bolt://localhost:7687",
                  user="neo4j", pwd="neo4j")
    conn.query("CREATE OR REPLACE DATABASE scratchdb")
    nx_g = nx.MultiDiGraph()
    nx_g.add_node(0, visiteds=True)
    nx_g.add_node(1, visiteds=False)
    nx_g.add_node(2, visiteds=True)
    nx_g.add_node(3, visiteds=False)
    nx_g.add_node(4, visiteds=True)
    nx_g.add_node(5, visiteds=False)
    nx_g.add_edge(0, 1)
    nx_g.add_edge(0, 2)
    nx_g.add_edge(4, 0)
    nx_g.add_edge(5, 4)
    nx_g.add_edge(4, 5)
    create_query = nx2create(0, nx_g)
    print(create_query)
    conn.query(create_query, db='scratchdb')
    nx_q = nx.MultiDiGraph()
    nx_q.add_node(0, visiteds=True)
    nx_q.add_node(1, visiteds=False)
    nx_q.add_edge(0, 1)

    # Normal match
    match_query = nx2query(nx_q)
    print(match_query)
    match_result = conn.query(match_query, db='scratchdb')
    print(match_result)
    # Each record is an instance
    # that can also be converted into a graph
    out_graphs = match2nx(nx_q, match_result)
    print(out_graphs)


    ## Let perform Out-of-structure query
    match_query_oos = nx2query_oos(nx_q)
    print("OOS query: ", match_query_oos)
    match_result = conn.query(match_query_oos, db='scratchdb')
    print('===================\n'*5)
    print(match_result)
    print('===================\n'*5)
    outputs = match_oos2nx(nx_q, match_result)
    print(outputs)
