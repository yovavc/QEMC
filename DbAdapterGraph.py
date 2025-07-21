class DbAdapterGraph:
    cur = None
    experiment_id = 0
    iteration_id = 0
    conn = None

    def __init__(self):
        import mariadb
        import sys

        # Connect to MariaDB Platform
        try:
            self.conn = mariadb.connect(
                user="root",
                password="root",
                host="localhost",
                port=3306,
                database="maxcut"

            )
            self.conn.autocommit = True
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

        # Get Cursor
        self.cur = self.conn.cursor()

    def get_experiments_graph_numberoflayer_expected_blacks(self):
        self.cur.execute(
            "select group_concat(id) as eId, graph, numberoflayer, expected_blacks, count(*) as c from experiments group by graph, numberoflayer, expected_blacks having c >= 10",
        )
        rv = []
        # Print Result-set
        for (eid, graph, numberoflayer, expected_blacks, c) in self.cur:
            rv.append({"eid": eid, "graph": graph, "numberOfLayers": numberoflayer, "expectedBlacks": expected_blacks,
                       "count": c})
        return rv

    def get_experiments_graph_numberoflayer_expected_blacks_by_node_count(self, nodeCount, layers, stepsize):
        self.cur.execute(
            "SELECT \
                GROUP_CONCAT(id) AS ids,\
                COUNT(*) AS count,\
                graphNodeCount,\
                numberOfLayer,\
                stepSize,\
                expected_blacks\
            FROM\
                maxcut.experiments\
            WHERE\
                ABS(stepSize - ?) < 1E-6\
                    AND graphNodeCount = ?\
                    AND numberOfLayer = ?\
                    AND valid = 1\
            GROUP BY graphNodeCount , numberOfLayer , stepSize , expected_blacks", (stepsize, nodeCount, layers)
        )

        rv = []
        # Print Result-set
        for (ids, count, graphNodeCount, numberOfLayer, stepSize, expected_blacks) in self.cur:
            rv.append({"ids": ids, "count": count, "graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer,
                        "stepSize": stepSize, "expected_blacks": expected_blacks})
        return rv

    def get_experiments_graph_numberoflayer(self):
        self.cur.execute(
            "select group_concat(id) as eId, graph, numberoflayer, expected_blacks, count(*) as c from experiments group by graph, numberoflayer having c >= 10",
        )
        rv = []
        # Print Result-set
        for (eid, graph, numberoflayer, expected_blacks, c) in self.cur:
            rv.append({"eid": eid, "graph": graph, "numberOfLayers": numberoflayer, "expectedBlacks": expected_blacks,
                       "count": c})
        return rv

    def get_single_experiment_maxcut(self, e_id):
        self.cur.execute(
            "select iterationIndex, value from iterations left join maxcut.maxcut on `maxcut`.`iterationId` = `iterations`.`id` where iterations.experimentId = ? order by iterations.iterationIndex",
            (e_id,))
        rv = []
        for (iterationIndex, maxcut) in self.cur:
            rv.append({"iterationIndex": iterationIndex, "maxcut": maxcut})
        return rv

    def get_maxcut_result_set(self, ids):

        self.cur.execute(
            "select max(`experiments`. `graph`) as graph, max(`iterations`. `iterationIndex`) as iterationIndex, avg(`maxcut_max`. `value`) as maxcutAvg, max(`maxcut_max`. `value`) as maxcutMax, min(`maxcut_max`. `value`) as maxcutMin from iterations left join maxcut.maxcut_max on `maxcut_max`. `iterationId` = `iterations`. `id` left join experiments on `iterations`. `experimentId` = `experiments`. `id` where iterations.experimentId in (" + ids + ") and experiments.valid = 1 group by iterationIndex order by iterationIndex"
        )
        rv = []
        for (graph, iterationIndex, maxcutAvg, maxcutMax, maxcutMin) in self.cur:
            rv.append({"graph": graph, "iteration": iterationIndex, "averageMaxcut": maxcutAvg, "maxMaxcut": maxcutMax,
                       "minMaxcut": maxcutMin})
        return rv

    def get_maxcut_instance_result_set(self, ids):

        self.cur.execute(
            "select max(`experiments`. `graph`) as graph, max(`iterations`. `iterationIndex`) as iterationIndex, avg(`maxcut`. `value`) as maxcutAvg, max(`maxcut`. `value`) as maxcutMax, min(`maxcut`. `value`) as maxcutMin from iterations left join maxcut.maxcut on `maxcut`. `iterationId` = `iterations`. `id` left join experiments on `iterations`. `experimentId` = `experiments`. `id` where iterations.experimentId in (" + ids + ") group by iterationIndex order by iterationIndex"
        )
        rv = []
        for (graph, iterationIndex, maxcutAvg, maxcutMax, maxcutMin) in self.cur:
            rv.append({"graph": graph, "iteration": iterationIndex, "averageMaxcut": maxcutAvg, "maxMaxcut": maxcutMax,
                       "minMaxcut": maxcutMin})
        return rv

    def get_maxcut_result_set1000(self, ids):

        self.cur.execute(
            "select max(`experiments`. `graph`) as graph, max(`iterations`. `iterationIndex`) as iterationIndex, avg(`maxcut_max_1000`. `value`) as maxcutAvg, max(`maxcut_max_1000`. `value`) as maxcutMax, min(`maxcut_max_1000`. `value`) as maxcutMin from iterations left join maxcut.maxcut_max_1000 on `maxcut_max_1000`. `iterationId` = `iterations`. `id` left join experiments on `iterations`. `experimentId` = `experiments`. `id` where iterations.experimentId in (" + ids + ") and experiments.valid = 1 group by iterationIndex order by iterationIndex"
        )
        rv = []
        for (graph, iterationIndex, maxcutAvg, maxcutMax, maxcutMin) in self.cur:
            rv.append({"graph": graph, "iteration": iterationIndex, "averageMaxcut": maxcutAvg, "maxMaxcut": maxcutMax,
                       "minMaxcut": maxcutMin})
        return rv

    def get_loss_result_set(self, ids):

        self.cur.execute(
            "SELECT \
                MAX(`experiments`.`graph`) AS graph,\
                MAX(`iterations`.`iterationIndex`) AS iterationIndex,\
                AVG(`loss`.`value`) AS lossAvg,\
                MAX(`loss`.`value`) AS lossMax,\
                MIN(`loss`.`value`) AS lossMin\
            FROM\
                iterations\
                    LEFT JOIN\
                maxcut.loss ON `loss`.`iterationId` = `iterations`.`id`\
                    LEFT JOIN\
                experiments ON `iterations`.`experimentId` = `experiments`.`id`\
            WHERE\
                iterations.experimentId IN (" + ids + ")\
                    AND experiments.valid = 1\
            GROUP BY iterationIndex\
            ORDER BY iterationIndex"
        )
        rv = []
        for (graph, iterationIndex, lossAvg, lossMax, lossMin) in self.cur:
            rv.append({"graph": graph, "iteration": iterationIndex, "averageLoss": lossAvg, "maxLoss": lossMax,
                       "minLoss": lossMin})
        return rv

    def get_max_maxcut_result_set(self, ids):
        self.cur.execute(
            f"SELECT \
                graph AS graph, MAX(avgMaxcut) as maxAvgMaxcut, layers, stepSize\
            FROM\
                (SELECT\
                    MAX(`experiments`.`graphNodeCount`) AS graph,\
                        MAX(`experiments`.numberOfLayer) AS layers,\
                        MAX(`experiments`.stepSize) AS stepSize,\
                        iterationIndex,\
                        AVG(maxcut_max.value) AS avgMaxcut\
                FROM\
                    iterations\
                LEFT JOIN maxcut_max ON maxcut_max.iterationId = iterations.id\
                LEFT JOIN experiments ON `iterations`.`experimentId` = `experiments`.`id`\
                WHERE\
                    experimentId IN ({ids})\
                GROUP BY iterationIndex) AS qty"
        )
        rv = []
        for (graph, maxAvgMaxcut, layers, stepSize) in self.cur:
            rv.append({"graph": graph, "maxAvgMaxcut": maxAvgMaxcut, "layers": layers, "stepSize": stepSize})
        return rv
    
    def get_max_maxcut_result_set_1000(self, ids):
        self.cur.execute(
            f"SELECT \
                graph AS graph, MAX(avgMaxcut) as maxAvgMaxcut, layers, stepSize\
            FROM\
                (SELECT\
                    MAX(`experiments`.`graphNodeCount`) AS graph,\
                        MAX(`experiments`.numberOfLayer) AS layers,\
                        MAX(`experiments`.stepSize) AS stepSize,\
                        iterationIndex,\
                        AVG(maxcut_max_1000.value) AS avgMaxcut\
                FROM\
                    iterations\
                LEFT JOIN maxcut_max_1000 ON maxcut_max_1000.iterationId = iterations.id\
                LEFT JOIN experiments ON `iterations`.`experimentId` = `experiments`.`id`\
                WHERE\
                    experimentId IN ({ids})\
                GROUP BY iterationIndex) AS qty"
        )
        rv = []
        for (graph, maxAvgMaxcut, layers, stepSize) in self.cur:
            rv.append({"graph": graph, "maxAvgMaxcut": maxAvgMaxcut, "layers": layers, "stepSize": stepSize})
        return rv


    def get_max_maxcut_result_set_expected_blacks(self, ids):
        self.cur.execute(
            f"SELECT \
            iterationIndex,\
            AVG(maxcut_max.value) AS avg,\
            STD(maxcut_max.value) AS std\
        FROM\
            iterations\
                LEFT JOIN\
            experiments ON iterations.experimentId = experiments.id\
                LEFT JOIN\
            maxcut_max ON maxcut_max.iterationId = iterations.id\
        WHERE\
            experiments.id IN (?)\
        GROUP BY iterations.iterationIndex"
            , (ids,))
        rv = []
        for (iterationIndex, avg, std) in self.cur:
            rv.append({"index": iterationIndex, "avg": avg, "std": std})
        return rv

    def get_experiments_group_by_stepsize_layers_graph(self):
        self.cur.execute(
            "SELECT group_concat(id) as ids, count(*) as c, numberOfLayer, stepSize, graph, graphNodeCount FROM maxcut.experiments where  numberOfLayer > 5 and valid =1 group by numberOfLayer, stepSize, graph order by graph, numberOfLayer"
        )
        rv = []
        for (ids, c, layers, stepSize, graph, graphNodeCount) in self.cur:
            rv.append({"ids": ids, "count": c, "layers": layers, "stepSize": stepSize, "graph": graph,
                       "graphNodeCount": graphNodeCount})
        return rv

    def get_experiments_group_by_stepsize_layers_graph_by_nodes(self, nodes):
        self.cur.execute(
            "SELECT group_concat(id) as ids, count(*) as c, numberOfLayer, stepSize, graph, graphNodeCount FROM maxcut.experiments where graphNodeCount=? and expected_blacks = graphNodeCount/2  and valid =1  and abs(stepSize -0.13) > 1e-6 and abs(stepSize -0.15) > 1e-6 group by numberOfLayer, stepSize, graphNodeCount order by graphNodeCount, numberOfLayer", (nodes,)
        )
        rv = []
        for (ids, c, layers, stepSize, graph, graphNodeCount) in self.cur:
            rv.append({"ids": ids, "count": c, "layers": layers, "stepSize": stepSize, "graph": graph,
                       "graphNodeCount": graphNodeCount})
        return rv

    def get_experiments_group_by_stepsize_layers_graph_by_nodes_1000(self, nodes):
        self.cur.execute(
            "SELECT group_concat(id) as ids, count(*) as c, numberOfLayer, stepSize, graph, graphNodeCount FROM maxcut.experiments where graphNodeCount=? and expected_blacks = graphNodeCount/2  and valid =1  and numberOfSteps = 1000 and abs(stepSize -0.13) > 1e-6 and abs(stepSize -0.15) > 1e-6 group by numberOfLayer, stepSize, graphNodeCount order by graphNodeCount, numberOfLayer", (nodes,)
        )
        rv = []
        for (ids, c, layers, stepSize, graph, graphNodeCount) in self.cur:
            rv.append({"ids": ids, "count": c, "layers": layers, "stepSize": stepSize, "graph": graph,
                       "graphNodeCount": graphNodeCount})
        return rv

    def get_maximum_maxcut_group_by_node_number(self):
        self.cur.execute(
            "select graphNodeCount, max(maxcut_max.value) as globalMaxMaxcut from iterations left join maxcut_max on maxcut_max.iterationId = iterations.id left join experiments on experiments.id = iterations.experimentId where iterations.iterationIndex =200 and expected_blacks= experiments.graphNodeCount/ 2 group by experiments.graphNodeCount order by graphNodeCount;"
        )
        rv = []
        for (graphNodeCount, globalMaxMaxcut) in self.cur:
            rv.append({"graphNodeCount": graphNodeCount, "globalMaxMaxcut": globalMaxMaxcut})
        return rv

    def get_avg_maxcut_group_by_node_number(self):
        self.cur.execute(
            "select graphNodeCount, avg(maxcut_max.value) as globalMaxMaxcut from iterations left join maxcut_max on maxcut_max.iterationId = iterations.id left join experiments on experiments.id = iterations.experimentId where iterations.iterationIndex =200 group by experiments.graphNodeCount order by graphNodeCount;"
        )
        rv = []
        for (graphNodeCount, globalMaxMaxcut) in self.cur:
            rv.append({"graphNodeCount": graphNodeCount, "globalMaxMaxcut": globalMaxMaxcut})
        return rv

    def get_experiments_group_by_stepsize_layers_graph_1024(self, nodes):
        self.cur.execute(
            "SELECT group_concat(id) as ids, count(*) as c, numberOfLayer, stepSize, graph, graphNodeCount FROM maxcut.experiments where graphNodeCount = ?  and valid =1 group by graphNodeCount , numberOfLayer, stepSize order by numberOfLayer",(nodes,)
        )
        rv = []
        for (ids, c, layers, stepSize, graph, graphNodeCount) in self.cur:
            rv.append({"ids": ids, "count": c, "layers": layers, "stepSize": stepSize, "graph": graph,
                       "graphNodeCount": graphNodeCount})
        return rv

    def get_experiments_group_by_stepsize_layers_graph_beta1(self):
        self.cur.execute(
            "select group_concat(id) as ids, count(*) as c, numberOfLayer, stepSize, graph, beta1  from maxcut.experiments where graph like \"%regular_graphs2%\" and time > '2022-04-25 08:01:53' group by numberOfLayer, stepSize, graph, beta1 order by graph, numberOfLayer"
        )
        rv = []
        for (ids, c, layers, stepSize, graph, beta1) in self.cur:
            rv.append({"ids": ids, "count": c, "layers": layers, "stepSize": stepSize, "graph": graph, "beta1": beta1})
        return rv

    def get_all_stepsize(self):
        self.cur.execute(
            "select stepSize from experiments group by stepSize order by stepSize;"
        )
        rv = []
        for (stepSize,) in self.cur:
            rv.append(stepSize)
        return rv

    def get_all_layers(self):
        self.cur.execute(
            "select numberOfLayer from experiments  group by numberOfLayer order by numberOfLayer"
        )
        rv = []
        for (stepSize,) in self.cur:
            rv.append(stepSize)
        return rv

    def experiments_by_layer_stepsize2(self, layers, stepSize, nodeCount):
        self.cur.execute(
            "select group_concat(experiments.id) as ids, count(*) as c, stepSize, numberOfLayer from experiments where numberOfLayer =? and graphNodeCount= ? and expected_blacks = graphNodeCount/2 and abs(stepSize -?) <= 1e-6 ",
            (layers, nodeCount, stepSize)
        )
        rv = []
        for (ids, c, stepSize, numberOfLayer) in self.cur:
            rv.append({"ids": ids, "count": c, "stepSize": stepSize, "numberOfLayer": numberOfLayer})
        return rv

    def absolute_max(self, nodes):
        self.cur.execute(
            "SELECT \
                graphNodeCount, numberOfLayer, stepSize, value\
            FROM\
                iterations\
                    LEFT JOIN\
                experiments ON experiments.id = iterations.experimentId\
                left join \
                maxcut_max on maxcut_max.iterationId = iterations.id\
            WHERE\
                graphNodeCount = ?\
                    and maxcut_max.value = (\
            SELECT\
                max(maxcut_max.value)\
            FROM\
                iterations\
                    LEFT JOIN\
                experiments ON experiments.id = iterations.experimentId\
                left join\
                maxcut_max on maxcut_max.iterationId = iterations.id\
            WHERE\
                graphNodeCount = ?\
                    AND expected_blacks = graphNodeCount / 2\
                    AND ABS(beta1 - 0.9) < 1E-6\
                    AND ABS(beta2 - 0.99) < 1E-6) order by numberOfLayer",
                        (nodes, nodes)
        )
        rv = []
        for (graphNodeCount, numberOfLayer, stepSize, value) in self.cur:
            rv.append({"graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer, "stepSize": stepSize, "value": value})
        return rv

    def absolute_max1000(self, nodes):
        self.cur.execute(
            "SELECT \
                graphNodeCount, numberOfLayer, stepSize, value\
            FROM\
                iterations\
                    LEFT JOIN\
                experiments ON experiments.id = iterations.experimentId\
                left join \
                maxcut_max_1000 on maxcut_max_1000.iterationId = iterations.id\
            WHERE\
                graphNodeCount = ?\
                    and maxcut_max_1000.value = (\
            SELECT\
                max(maxcut_max_1000.value)\
            FROM\
                iterations\
                    LEFT JOIN\
                experiments ON experiments.id = iterations.experimentId\
                left join\
                maxcut_max_1000 on maxcut_max_1000.iterationId = iterations.id\
            WHERE\
                graphNodeCount = ?\
                    AND expected_blacks = graphNodeCount / 2\
                    AND ABS(beta1 - 0.9) < 1E-6\
                    AND ABS(beta2 - 0.99) < 1E-6) order by numberOfLayer",
                        (nodes, nodes)
        )
        rv = []
        for (graphNodeCount, numberOfLayer, stepSize, value) in self.cur:
            rv.append({"graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer, "stepSize": stepSize, "value": value})
        return rv

    def get_time(self, nodes, layers):
        self.cur.execute(
            "SELECT avg(iterations.duration) as duration FROM maxcut.iterations \
                                left join experiments on  experiments.id = iterations.experimentId \
                                where duration <> 0 and experimentId > 42098 and numberOfLayer =? and graphNodeCount = ?\
                                group by  graphNodeCount, numberOfLayer\
                                order by iterations.id desc", (layers, nodes))
        rv = []
        for (duration) in self.cur:
            rv.append(duration)

        return rv[0]
