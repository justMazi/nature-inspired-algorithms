import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    import pydot
except ImportError:
    pydot = None

from IPython.display import Image, display


def plot_stats(
    statistics,
    ylog=False,
    view=True,
    filename=None,
    fmt="png"
):
    """ Plots the population's average and best fitness. """
    
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, "b-", label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, "g-.", label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, "g-.", label="+1 sd")
    plt.plot(generation, best_fitness, "r-", label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale("symlog")

    if filename:
        plt.savefig(filename + "." + fmt)
        
    if view:
        plt.show()

    plt.close()


def plot_species(
    statistics,
    view=True,
    filename=None,
    fmt="png"
):
    """ Visualizes speciation throughout evolution. """
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    if filename:
        plt.savefig(filename + "." + fmt)
        
    if view:
        plt.show()

    plt.close()


def draw_net(
    config,
    genome,
    view=True,
    filename=None,
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    node_colors=None,
    fmt="png"
):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    
    if pydot is None:
        warnings.warn("This display is not available due to a missing optional dependency (pydot)")
        return

    # Attributes for network nodes.
    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict
    
    node_attrs = {
        "shape": "circle",
        "fontsize": "9",
        "height": "0.2",
        "width": "0.2"}

    # Create the graph
    graph = pydot.Dot(graph_type="digraph", **node_attrs)

    # Add input nodes to the graph
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_node_attrs = {"style": "filled", "shape": "box", "fillcolor": node_colors.get(k, "lightgray")}
        
        node = pydot.Node(name, **input_node_attrs)
        graph.add_node(node)

    # Add output nodes to the graph
    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        output_node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}

        node = pydot.Node(name, **output_node_attrs)
        graph.add_node(node)

    # Prune the graph (do not render unused nodes), if required
    if prune_unused:
        # First, find all the connections being actively used by the network (Or take all, if required)
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input_node_id, output_node_id = cg.key[0], cg.key[1]
                connections.add((input_node_id, output_node_id))

        # Perform search (basically a BFS) from the output nodes using the connections we found above
        # to see what nodes are used when computing the outputs.
        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for input_node_id, output_node_id in connections:
                if output_node_id in pending and input_node_id not in used_nodes:
                    new_pending.add(input_node_id)
                    used_nodes.add(input_node_id)
            pending = new_pending
            
    # Else render all the nodes
    else:
        used_nodes = set(genome.nodes.keys())
        
    # Add the (possibly just used) inner nodes to the graph
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        name = str(n)
        inner_node_attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        
        node = pydot.Node(name, **inner_node_attrs)
        graph.add_node(node)
    
    # Add the connections to the graph
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_node_id, output_node_id = cg.key
            
            # Do not render pruned connections
            if prune_unused and (input_node_id not in used_nodes or output_node_id not in used_nodes):
               continue
           
            input_node_name = node_names.get(input_node_id, str(input_node_id))
            output_node_name = node_names.get(output_node_id, str(output_node_id))
            edge_attributes = {
                "style": "solid" if cg.enabled else "dotted",
                "color": "green" if cg.weight > 0 else "red",
                "penwidth": str(0.1 + abs(cg.weight / 5.0))
            }
            
            edge = pydot.Edge(input_node_name, output_node_name, **edge_attributes)
            graph.add_edge(edge)

    # Display the graph, if required
    if view:
        display(Image(graph.create(format=fmt)))
        
    # Save the graph to file, if required
    if filename:
        graph.write(filename + "." + fmt, format=fmt)

    return graph