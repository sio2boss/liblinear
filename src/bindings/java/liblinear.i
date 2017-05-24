/*
 * liblinear.i
 */

%module liblinear

%{
#include "linear.h"
#include "io.h"
%}
 
%rename(Parameter) parameter;
%rename(Problem) problem;
%rename(Model) model;
%rename(FeatureNode) feature_node;
%rename(Liblinear) liblinear;

%include linear.h
%include io.h