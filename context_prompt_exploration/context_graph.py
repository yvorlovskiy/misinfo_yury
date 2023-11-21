from graphviz import Digraph

# Define the groups and sub-groups
groups = {
    "Evidence and Verification": {
        "Visual Evidence/Verification": ["(video, evidence)", "(photo, evidence)", "(image, evidence)", "(image, verification)", "(photo, verification)"],
        "Audio Evidence": ["(audio, evidence)"],
        "Statistical Evidence/Verification": ["(statistical, evidence)", "(crime, statistics)", "(job, statistics)", "(voter, statistics)", "(vaccination, statistics)", "(economic, data)", "(inflation, rate)", "(salary, data)", "(price, data)"],
        "Policy and Legislative Evidence/Verification": ["(policy, evidence)", "(policy, specifics)", "(legislation, evidence)", "(tax, legislation)", "(current, legislation)", "(bill, number)"],
        "Source Verification": ["(source, verification)", "(verification, source)", "(source, credibility)", "(source, reliability)", "(news, source)", "(data, source)", "(confirmation, source)"]
    },
    "Date and Time Context": {
        "": ["(event, date)", "(tweet, date)", "(publication, date)", "(broadcast, date)", "(speech, date)"]
    },
    "Miscellaneous Context": {
        "": ["(event, context)", "(speech, context)", "(contextual, evidence)", "(historical, evidence)", "(current, evidence)", "(photograph, evidence)"]
    }
}

# Create a new directed graph
dot = Digraph(format='png', engine='dot')

# Add nodes and edges to the graph
for group, subgroups in groups.items():
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node(group)
        for subgroup, items in subgroups.items():
            if subgroup:  # only add subgroup if it's not an empty string
                s.node(subgroup)
                s.edge(group, subgroup)
            item_list = ', '.join(items)
            item_node_name = f"{subgroup if subgroup else group}_items"
            s.node(item_node_name, label=item_list, shape='record')
            s.edge(subgroup if subgroup else group, item_node_name)

# Render and view the graph
dot.render('output', view=True)