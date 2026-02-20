
        # 1. Create agents and place them on nodes
        for i, node in enumerate(self.G.nodes):
            if i >= self.num_agents:
                break  # Safety check if graph has more nodes than num_agents
            attributes = self.G.nodes[node]
            agent = FirmAgent(
                self,
                sector=attributes["sector"],
                postcode=attributes["postcode"],
                network=attributes["network"],
                size=attributes["size"]
            )

            self.grid.place_agent(agent, node) # The agent is anchored to a node in Mesa's network grid, and get_neighbors() gives it access to peers via edges.]

        #2. Build network from these agents
        self.G = self.build_network()  # Create your network (spatial + business links)
        #3. Create a grid fro mthe network
        self.grid = NetworkGrid(self.G)  # Place agents on a network grid
        #4. Place agents on nodes

        #5. debug and collect data
        print("Num agents created:", self.num_agents)
        print("Num graph nodes:", self.G.number_of_nodes())
        print("Num graph edges:", self.G.number_of_edges())
        print("Agents placed on grid:", len(self.grid.get_all_cell_contents()))

        # Create a DataCollector to collect data for analysis
        self.datacollector = DataCollector(
            model_reporters={
                "Num_Developers": lambda m: m.count_adoption_stage("C. Is developing a WTP"), # Count (using a helper function) how many agents are in the 'developing a WTP' stage
                "Num_Adopters": lambda m: ( 
                    m.count_adoption_stage("D. Has a low efficacy WTP") # Count how many agents have a poorly implemented WTP
                    + m.count_adoption_stage("E. Has an effective WTP") # Count (and add) how many agents have a well implemented WTP
                    )
            },
            agent_reporters={
                # Add other variables as needed for debugging/plotting
                "Adoption Stage": "adoption_stage",
                "Adoption Probability": "prob_adoption"
            }
        ) # This is collecting a lot of data and causing bloat, may need to find a way to reduce this.

        self.datacollector.collect(self)  # Collect initial state (step 0)

    def step(self): # Now we are defining the methods, the things the model can do. This one advancces time
        """Advance the model by one step."""
        self.apply_active_shocks()         # Apply exogenous shocks
        # Activate agents
        self.agents.shuffle_do("step")  # Random activation of all agents
        self.datacollector.collect(self)  # Collect data after each step


    def build_network(self): # This is a helper methods, to construct my structure
        """ This creates a NetworkX graph where each node represents a firm and edges represent potential social influence."""
        G = nx.Graph()  # Create an empty undirected graph

        for a in self.agents:  # for all my agents, add a node to the graph, inclduing their attributes https://memgraph.github.io/networkx-guide/basics/#reading-graphs
            G.add_node(
                a.unique_id,
                postcode=a.postcode,
                sector=a.sector,
                network=a.network,
                size=a.size,
            )
        
        # Group ids by linkage keys
        by_postcode = {}
        by_network = {}

        for a in self.agents:
            by_postcode.setdefault(a.postcode, []).append(a.unique_id)
            if a.network is not None and a.network != "None":
                by_network.setdefault(a.network, []).append(a.unique_id)

    # Helper to add edges within a group
    def connect_group(id_list, reason):
        # reason is either "spatial" if they are in the same postcode or "network" if they are in the same network
        for i, j in itertools.combinations(id_list, 2):
            p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
            p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"]

            spatial_link = (p1 == p2) # If they are in the same postcode, then spatial link = TRUE
            network_link = (n1 is not None and n1 != "None" and n1 == n2) # If they are in the same network, then they have a network link

            if not (spatial_link or network_link):
                continue

            # Your existing edge type logic
            if (spatial_link and s1 != s2) or network_link: # If they are in network, or same geography and different sectors, then they are peers. Otherwise competitor
                edge_type = "peer"
            else:
                edge_type = "competitor"

            # Avoid overwriting an existing edge type in a way that loses information
            if G.has_edge(i, j):
                existing = G.edges[i, j].get("type")
                if existing != edge_type:
                    G.edges[i, j]["type"] = "peer"
                    G.edges[i, j]["types"] = sorted(set([existing, edge_type]))
                else:
                    G.add_edge(i, j, type=edge_type)
        return G

    # Connect within each postcode clique
    for ids in by_postcode.values():
        if len(ids) > 1:
            connect_group(ids, reason="spatial")

    # Connect within each business network clique
    for ids in by_network.values():
        if len(ids) > 1:
            connect_group(ids, reason="network")

    print("Agents in schedule:", len(self.schedule.agents))
    print("Agents on grid:", len(self.grid.get_all_cell_contents()))

        
        # # Now we add edges based on the attributes of the nodes
        # for i in G.nodes:
        #     for j in G.nodes:
        #         if i >= j:
        #             continue  # Avoid self-loops and duplicates

        #         p1, s1, n1 = G.nodes[i]["postcode"], G.nodes[i]["sector"], G.nodes[i]["network"]
        #         p2, s2, n2 = G.nodes[j]["postcode"], G.nodes[j]["sector"], G.nodes[j]["network"] # This may cause issues because each edge uses at least 100 bytes of memory https://memgraph.github.io/networkx-guide/faq#load-graph

        #         spatial_link = p1 == p2  # Firms are co-located
        #         network_link = n1 != "None" and n1 == n2  # Firms are in same business network

        #         if spatial_link or network_link:
        #             # Edge type logic
        #             if (spatial_link and s1 != s2) or network_link:
        #                 edge_type = "peer"
        #             else:
        #                 edge_type = "competitor"
        #             G.add_edge(i, j, type=edge_type)  # Add edge with type metadata

        # return G