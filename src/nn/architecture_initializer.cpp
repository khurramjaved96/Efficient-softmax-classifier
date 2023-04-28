//
// Created by Khurram Javed on 2023-03-18.
//

#include "../../include/nn/architure_initializer.h"
#include <iostream>
#include <random>

//Graph *ArchitectureInitializer::initialize_sprase_networks(
//
//    Graph *my_graph, int total_parameters, int parameters_per_feature,
//    std::string vertex_type, float step_size) {
//  std::mt19937 mt;
//  std::uniform_real_distribution<float> weight_sampler(-0.02, 0.02);
//  int input_features = my_graph->list_of_vertices.size();
//  for (int i = 0; i < 3; i++) {
//    my_graph->list_of_vertices.push_back(
//        VertexFactory::get_vertex(vertex_type));
//    for (int j = 0; j < input_features; j++) {
//      my_graph->add_edge(weight_sampler(mt), j,
//                         my_graph->list_of_vertices.size() - 1, step_size);
//    }
//  }
//  int total_vertices_features = my_graph->list_of_vertices.size();
//  for (int i = 0; i < my_graph->output_vertices; i++) {
//    Vertex *output_vertex = VertexFactory::get_vertex("linear");
//    output_vertex->is_output = true;
//    my_graph->list_of_vertices.push_back(output_vertex);
//    for (int j = input_features; j < total_vertices_features; j++) {
//      my_graph->add_edge(weight_sampler(mt), j,
//                         my_graph->list_of_vertices.size() - 1, step_size);
//    }
//  }
//
//  return my_graph;
//}

Graph *ArchitectureInitializer::initialize_sprase_networks(
    Graph *my_graph, int total_parameters, int parameters_per_feature,
    std::string vertex_type, float step_size) {

  int total_features = int(total_parameters / parameters_per_feature);
  for (int i = 0; i < total_features; i++) {
    my_graph->list_of_vertices.push_back(
        VertexFactory::get_vertex(vertex_type));
    auto sampler = std::uniform_int_distribution<int>(
        0, my_graph->list_of_vertices.size() - 2);
    for (int j = 0; j < parameters_per_feature; j++) {
      my_graph->add_edge(0, sampler(my_graph->mt),
                         my_graph->list_of_vertices.size() - 1, step_size);
    }
  }
  int total_vertices_features = my_graph->list_of_vertices.size();
  for (int i = 0; i < my_graph->output_vertices; i++) {
    Vertex *output_vertex = VertexFactory::get_vertex("linear");
    output_vertex->is_output = true;
    my_graph->list_of_vertices.push_back(output_vertex);
    for (int j = 0; j < total_vertices_features; j++) {
      my_graph->add_edge(0, j, my_graph->list_of_vertices.size() - 1,
                         step_size);
    }
  }
  return my_graph;
}