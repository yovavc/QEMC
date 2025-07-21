class DbAdapter:
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

    def new_experiment(self, graph, expected_blacks, node_count, number_of_layers, is_entangled,
                       number_of_steps, init, comment="", step_size=0.2, beta1=0.9, beta2=0.99, eps=1e-08,  shots=-1, backend="sym"):
        self.cur.execute(
            "INSERT INTO `maxcut`.`experiments` (`graph`, `expected_blacks`, `graphNodeCount`, `numberOfLayer`, `isEntangled`, `comment`, `stepSize`, `beta1`, `beta2`, `epsilon`, `numberOfSteps`, `init`, shots, backend) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (graph, expected_blacks, node_count, number_of_layers, is_entangled, comment, step_size,
             beta1, beta2, eps, number_of_steps, init, shots, backend))
        self.experiment_id = self.cur.lastrowid

    def add_iteration(self, index):
        self.cur.execute("INSERT INTO iterations (`experimentId`, `iterationIndex`) VALUES (?,?)", (self.experiment_id, index))

        self.iteration_id = self.cur.lastrowid

    def update_iteration(self, lap_time):
        self.cur.execute("UPDATE iterations set duration=? where id = ?",
                         (lap_time, self.iteration_id))

    def add_loss(self, value):
        self.cur.execute("INSERT INTO loss (`value`, `iterationId`) VALUES (?,?)", (value, self.iteration_id))

    def add_muxcut(self, value):
        self.cur.execute("INSERT INTO maxcut (`value`, `iterationId`) VALUES (?,?)", (value, self.iteration_id))

    def add_group(self, value):
        self.cur.execute("INSERT INTO `groups` (`value`, `iterationId`) VALUES (?,?)", (value, self.iteration_id))

    def add_probs(self, value):
        self.cur.execute("INSERT INTO `probs` (`value`, `iterationId`) VALUES (?,?)", (value, self.iteration_id))

    def add_blacks(self, value):
        self.cur.execute("INSERT INTO `blacks` (`value`, `iterationId`) VALUES (?,?)", (value, self.iteration_id))

    def get_current_experiments_count(self, layers, step_size, node_count, steps):
        self.cur.execute(
            "SELECT \
                COUNT(*) AS count, graphNodeCount, numberOfLayer, stepSize\
            FROM\
                maxcut.experiments\
            WHERE\
                expected_blacks = graphNodeCount / 2\
                    AND ABS(stepSize - ?) < 1E-6\
                    AND graphNodeCount = ?\
                    AND numberOfLayer = ?\
                    AND valid =1\
                    AND numberOfSteps = ?\
            GROUP BY graphNodeCount , numberOfLayer , stepSize", (step_size, node_count, layers, steps)
        )
        rv = []
        for (count, graphNodeCount, numberOfLayer, stepSize) in self.cur:
            rv.append(
                {"count": count, "graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer, "stepSize": stepSize}
            )
        return rv

    def get_current_experiments_count_expected_blacks(self, layers, step_size, node_count, expected_blacks):
        self.cur.execute(
            "SELECT \
                COUNT(*) AS count, graphNodeCount, numberOfLayer, stepSize, expected_blacks\
            FROM\
                maxcut.experiments\
            WHERE\
                ABS(stepSize - ?) < 1E-6\
                    AND graphNodeCount = ?\
                    AND numberOfLayer = ?\
                    AND expected_blacks = ?\
                    AND valid =1", (step_size, node_count, layers, expected_blacks)
        )
        rv = []
        for (count, graphNodeCount, numberOfLayer, stepSize, expected_blacks) in self.cur:
            rv.append(
                {"count": count, "graphNodeCount": graphNodeCount, "numberOfLayer": numberOfLayer, "stepSize": stepSize, "expected_blacks": expected_blacks}
            )
        return rv
