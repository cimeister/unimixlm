#ifndef HIDDEN_LAYER_HH
#define HIDDEN_LAYER_HH

#include <vector>
#include <string>
#include "matrix.hh"
#include "utils.hh"
#include "layer.hh"

class HiddenLayer : public Layer {
public:
  HiddenLayer() = default;
  HiddenLayer(const HiddenLayer &) = default;
  HiddenLayer(HiddenLayer &&) = default;
  HiddenLayer & operator=(const HiddenLayer &) = default;
  HiddenLayer & operator=(HiddenLayer &&) = default;
  ~HiddenLayer() = default;
  HiddenLayer(int, int, std::string func="logistic");
  HiddenLayer(const Matrix &, std::string func="logistic");
  HiddenLayer(const Matrix &, const Matrix &, std::string func="logistic");
  Matrix output();
  Matrix output(const Matrix &); 
  Matrix forward();  
  Matrix forward(int);  
  Matrix forward(int, int);  
  Matrix forward(const Matrix &, std::string func="logistic");
  Matrix backward();
  Matrix backward(int);
  Matrix backward(int, int);
  Matrix backward(const Matrix &);  
  void update(double);
  void update(double, double);
  std::vector<std::size_t> shape() const;
private:
  Matrix weight;
  Matrix w_bias;
  Matrix delta_weight;  
  Matrix delta_w_bias;
  // Matrix x_forward;
  Matrix local_field;
  Matrix local_gradient;
  std::string activ_func_name;
};

#endif  // hidden_layer.hh