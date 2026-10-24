/**
  *   @file Vector2.cpp
  *   @author <NAME>
  *   @brief Source code for class Vector2
  */

#include "Vector2.hpp"

namespace pe {


  /*  Constructors */
  template<typename T>
  Vector2<T>::Vector2(T x, T y): x(x), y(y) {}

  template<typename T>
  Vector2<T>::Vector2(): x(static_cast<T> (0)), y(static_cast<T> (0)) {}

  template<typename T>
  Vector2<T>::Vector2(const Vector2<T>& vector2): x(vector2.x), y(vector2.y) {}

  template<typename T>
  T Vector2<T>::getX() const {
    return x;
  }

  template<typename T>
  T Vector2<T>::getY() const {
    return y;
  }

  template<typename T>
  void Vector2<T>::update(T x, T y) {
    this->x = x;
    this->y = y;
  }

  template<typename T>
  void Vector2<T>::normalize() {
    float norm = std::sqrt(x * x + y * y);
    if (norm) {
      x = x / norm;
      y = y / norm;
    }
  }

  template<typename T>
  float Vector2<T>::dotProduct(const Vector2<T>& vector) const {
    return x * vector.x + y * vector.y;
  }

  template<typename T>
  Vector2<T>& Vector2<T>::rotate(T angle) {
    if (angle != 0) {
      T sin = std::sin(angle);
      T cos = std::cos(angle);
      T x1 = x;
      T y1 = y;
      x = cos * x1 - sin * y1;
      y = sin * x1 + cos * y1;
    }
    return *this;
  }

  template<typename T>
  T Vector2<T>::getAngle() const {
    return std::atan2(y, x);
  }

  template<typename T>
  T Vector2<T>::getLength() const {
    return std::sqrt(x * x + y * y);
  }

  template<typename T>
  Vector2<T>& Vector2<T>::operator=(const Vector2<T>& vector2) {
    x = vector2.x;
    y = vector2.y;
    return *this;
  }

  template<typename T>
  Vector2<T>& Vector2<T>::operator*=(T mult) {
    x *= mult;
    y *= mult;
    return *this;
  }

  template<typename T>
  Vector2<T>& Vector2<T>::operator+=(const Vector2<T>& vector2) {
    x += vector2.x;
    y += vector2.y;
    return *this;
  }

  template<typename T>
  Vector2<T>& Vector2<T>::operator-=(const Vector2<T>& vector2) {
    x -= vector2.x;
    y -= vector2.y;
    return *this;
  }

  /*  Instantiate needed templates */
  template class Vector2<int>;
  template class Vector2<float>;
  template class Vector2<unsigned>;
  template class Vector2<double>;

} // end of namespace pe