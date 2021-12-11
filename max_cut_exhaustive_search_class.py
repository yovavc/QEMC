from tqdm import tqdm
import numpy as np
class mac_cut_exhaustive_search:
    def arcs_from_binary(self,arcs,binary_string):
        binary_arcs = []
        for arc in arcs:
            first_node = arc[0]
            second_node = arc[1]
            binary_arc = [int(binary_string[first_node]),int(binary_string[second_node])]
            binary_arcs.append(binary_arc)
        return binary_arcs

    def calc_max_cut_score(self,subset,arcs):
        binary_arcs = self.arcs_from_binary(arcs,subset)
        max_cut_score = 0
        for binary_arc in binary_arcs:
            if binary_arc == [0,1]:max_cut_score+=1
        return max_cut_score

    def get_best_group(self,scores):
        min_value = np.min(scores[:,2])
        min_value_indices = np.where(scores[:,2]==min_value)
        min_group_difference = len(scores[0,0])
        min_group_index = None
        for index in min_value_indices[0]:
            current_groups = scores[index,1]
            current_groups_difference = np.abs(len(current_groups[0])-len(current_groups[1]))
            if current_groups_difference<min_group_difference:
                min_group_difference = current_groups_difference
                min_group_index = index
        best_group = scores[min_group_index,1]
        return best_group,min_value

    def get_max_cut_group(self,arcs):
        scores = []
        num_of_nodes = (np.amax(np.array(arcs))+1)
        range_array = np.array(range(0,num_of_nodes))
        binary_format = '{0:0'+str(num_of_nodes)+'b}'
        for i in tqdm(range(1,2**int(num_of_nodes-1))):
            binary_string = binary_format.format(i)
            max_cut_score = self.calc_max_cut_score(binary_string, arcs)
            scores.append([range_array,binary_string,max_cut_score])
        scores = np.array(scores, dtype=object)
        max_cut_binary_string = scores[np.argmax(scores[:,2])][1]
        return max_cut_binary_string

        ### using example ###
        # graph = all_graphs[1]
        # current_arcs = list(graph.edges)
        # group, conn_nods_same_group = get_max_cut_group(current_arcs)
        ######
