from .ontology import get_onto_protein_uri, \
    get_onto_cells_by_protein, \
    get_onto_sub_classes, \
    get_onto_structures_by_related_uri, \
    get_onto_tissues_by_cell


def get_probe_structure_map(ontology, probe_labels):
    print("Building probe / structure map for %s" % " ".join(probe_labels))

    probe_uri_dict = {}
    probe_structure_dict = {}

    for label in probe_labels:
        label_str = label.replace('Anti-', '')
        uri_result = get_onto_protein_uri(ontology, label_str)
        if len(uri_result) == 0:
            continue
        probe_uri_dict[label] = uri_result[0][0]

    for label, protein_uri in probe_uri_dict.items():
        print(label)
        if label not in probe_structure_dict.keys():
            probe_structure_dict[label] = {'has_part': set(), 'surrounded_by': set()}

        cells = get_onto_cells_by_protein(ontology, protein_uri)

        sub_cells = []

        for cell in cells:
            sub_cells.extend(get_onto_sub_classes(ontology, cell[0]))

        cells.extend(sub_cells)

        for cell in cells:
            print('\t', cell[0].split('#')[1])

            # first check if the cell is directly related to a structure
            structures = get_onto_structures_by_related_uri(ontology, cell[0])

            if len(structures) > 0:
                for structure in structures:
                    print('\t\t\t', structure[0].split('#')[1])
                    rel_type = structure[2].split('#')[1]

                    probe_structure_dict[label][rel_type].add(structure[1].value)

                    sub_structs = get_onto_sub_classes(ontology, structure[0])
                    for ss in sub_structs:
                        print('\t\t\t\tsub_struct: ', ss[0].split('#')[1])
                        probe_structure_dict[label][rel_type].add(ss[1].value)

            tissues = get_onto_tissues_by_cell(ontology, cell[0])

            sub_tissues = []

            for tissue in tissues:
                sub_tissues.extend(get_onto_sub_classes(ontology, tissue[0]))

            tissues.extend(sub_tissues)

            for tissue in tissues:
                print('\t\t', tissue[0].split('#')[1])

                structures = get_onto_structures_by_related_uri(ontology, tissue[0])

                for structure in structures:
                    print('\t\t\t', structure[0].split('#')[1])
                    rel_type = structure[2].split('#')[1]

                    probe_structure_dict[label][rel_type].add(structure[1].value)

                    sub_structs = get_onto_sub_classes(ontology, structure[0])

                    for ss in sub_structs:
                        print('\t\t\t\tsub_struct: ', ss[0].split('#')[1])
                        probe_structure_dict[label][rel_type].add(ss[1].value)

    return probe_structure_dict


def build_seg_config(ontology, probes, probe_colors, cell_size, kernel_adjustment=0):
    probe_colors = [c.lower() for c in probe_colors]
    probe_structure_map = get_probe_structure_map(ontology, probes)
    k = kernel_adjustment

    has_part_colors = set()

    for i, p in enumerate(probes):
        ps_map = probe_structure_map[p]

        if len(ps_map['has_part']) > 0:
            has_part_colors.add(probe_colors[i])

    non_has_part_colors = set(probe_colors).difference(has_part_colors)

    # because DAPI is blue, probe colors can get slightly mixed with blue
    # so we'll add the blue-ish version of each color for better results
    if 'green' in has_part_colors:
        has_part_colors.add('cyan')
    if 'red' in has_part_colors:
        has_part_colors.add('violet')
    if 'white' in has_part_colors:
        has_part_colors.add('gray')

    if 'green' in non_has_part_colors:
        non_has_part_colors.add('cyan')
    if 'red' in non_has_part_colors:
        non_has_part_colors.add('violet')
    if 'white' in non_has_part_colors:
        non_has_part_colors.add('gray')

    seg_config = [
        # 1st seg stage uses "has_part" colors
        {
            'type': 'color',
            'args': {
                'blur_kernel': (15 + k, 15 + k),
                'min_size': 3 * cell_size,
                'max_size': None,
                'colors': has_part_colors
            }
        },
        # 2nd - 4th stages are saturation stages of descending sizes
        {
            'type': 'saturation',
            'args': {
                'blur_kernel': (63 + k, 63 + k),
                'min_size': 3 * cell_size,
                'max_size': None
            }
        },
        {
            'type': 'saturation',
            'args': {
                'blur_kernel': (31 + k, 31 + k),
                'min_size': 3 * cell_size,
                'max_size': None
            }
        },
        {
            'type': 'saturation',
            'args': {
                'blur_kernel': (15 + k, 15 + k),
                'min_size': 3 * cell_size,
                'max_size': None
            }
        },
        # final stage is a color stage on the "non_has_part" colors
        {
            'type': 'color',
            'args': {
                'blur_kernel': (7 + k, 7 + k),
                'min_size': 3 * cell_size,
                'max_size': None,
                'colors': non_has_part_colors
            }
        }
    ]

    return seg_config
