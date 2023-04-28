//
// Created by Khurram Javed on 2023-03-18.
//

#ifndef INCLUDE_NN_ARCHITURE_INITIALIZER_H_
#define INCLUDE_NN_ARCHITURE_INITIALIZER_H_
#include "networks/graph.h"
#include <string>

class ArchitectureInitializer{
public:
  ArchitectureInitializer() = default;
  Graph *initialize_sprase_networks(Graph *my_graph, int total_parameters,
                                    int parameters_per_feature,
                                    std::string vertex_type, float step_size);
};



#endif // INCLUDE_NN_ARCHITURE_INITIALIZER_H_
