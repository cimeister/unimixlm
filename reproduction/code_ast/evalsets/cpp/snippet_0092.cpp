//========= Copyright N11 Software, All rights reserved. ============//
//
// Purpose: ---
//
//===================================================================//
#include "idVec3.h"
#include "Vector.h"


idVec3::idVec3() 
{
	x = 0.0f;
	y = 0.0f;
    z = 0.0f;
}

idVec3::idVec3(const float& x, const float& y, const float& z) 
{
	this->x = x;
	this->y = y;
    this->z = z;
}

idVec3& idVec3::add(const idVec3& other) 
{
	x += other.x;
	y += other.y;
    z += other.z;

	return *this;
}

idVec3& idVec3::subtract(const idVec3& other) 
{
	x -= other.x;
	y -= other.y;
    z -= other.z;

	return *this;
}

idVec3& idVec3::multiply(const idVec3& other) 
{
	x *= other.x;
	y *= other.y;
    z *= other.z;

	return *this;
}

idVec3& idVec3::divide(const idVec3& other) 
{
	x /= other.x;
	y /= other.y;
    z /= other.z;

	return *this;
}

idVec3& operator+(idVec3 left, const idVec3& right) 
{
	return left.add(right);
}

idVec3& operator-(idVec3 left, const idVec3& right) 
{
	return left.subtract(right);
}

idVec3& operator*(idVec3 left, const idVec3& right) 
{
	return left.multiply(right);
}

idVec3& operator/(idVec3 left, const idVec3& right) 
{
	return left.divide(right);
}

idVec3 &idVec3::operator+=(const idVec3 &other) 
{
	return add(other);
}
idVec3 &idVec3::operator-=(const idVec3 &other) 
{
	return subtract(other);
}
idVec3 &idVec3::operator*=(const idVec3 &other) 
{
	return multiply(other);
}

idVec3 &idVec3::operator/=(const idVec3 &other) 
{
	return divide(other);
}

std::ostream& operator<<(std::ostream& stream, const idVec3& vector) 
{
	stream << "idVec3: (" << vector.x << ", " << vector.y << ", " << vector.z << ")";
	return stream;
}