#include <TinyAD/Scalar.hh>
#include <stdio.h>
#include <iostream>

// Choose autodiff scalar type for 3 variables
using ADouble = TinyAD::Double<3>;


int main() {
    // Init a 3D vector of active variables and a 3D vector of passive variables
    Eigen::Vector3<ADouble> x = ADouble::make_active({0.0, -1.0, 1.0});
    Eigen::Vector3<double> y(2.0, 3.0, 5.0);

    // Compute angle using Eigen functions and retrieve gradient and Hessian w.r.t. x
    ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));
    Eigen::Vector3d g = angle.grad;
    Eigen::Matrix3d H = angle.Hess;


    std::cout << "Angle: " << angle.val << std::endl;
    std::cout << "Gradient: " << g.transpose() << std::endl;
    return 0;
}