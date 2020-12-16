// Given left/right values and constraints, tweak left/right to optimize

#ifndef LRPATH_H
#define LRPATH_H

#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdio>
#include <boost/array.hpp>
const double pi = M_PI;
const double D_TO_R = pi/180.;


typedef boost::array<double, 3> array3;
typedef boost::array<double, 2> array2;

struct Lr_model {
    double bxl;
    double bxr;
    double qx;
    double byl;
    double byr;
    double qy;
    double bol;
    double bor;
    double qo;
    Lr_model():
        bxl{1.258}, bxr{1.378}, qx{7.929},
        byl{-0.677}, byr{0.657}, qy{5.650},
        bol{-7.659}, bor{7.624}, qo{8.464}
    {}
};

class Path
{
public:
    // class variables

    // dynamic model
    const Lr_model LRM;

    double bhxl, bhxr, bhyl, bhyr, bhol, bhor;
    double alphax, alphay, alphao;
    Eigen::ArrayXd alphaxj, alphayj, alphaoj, betaj;
    // 1st derivatives
    Eigen::ArrayXXd dpxdl, dpxdr, dpydl, dpydr;
    // 2nd derivatives
    Eigen::ArrayXXd d2pxdldl;
    Eigen::ArrayXXd d2pxdldr;
    Eigen::ArrayXXd d2pxdrdr;
    Eigen::ArrayXXd d2pydldl;
    Eigen::ArrayXXd d2pydldr;
    Eigen::ArrayXXd d2pydrdr;
    
    double dt;
    double lossValue;
    int n;

    Eigen::ArrayXd vxj, vyj, omegaj, pxj, pyj, thetaj;
    Eigen::ArrayXd lefts, rights;

    array3 start_pose, start_twist, target_pose, target_twist;
    array2 target_lr;
    double Wmax, Wjerk, Wback, mmax, converge_ratio;

    // intermediate values
    Eigen::ArrayXd cosj, sinj, vxcj, vxsj, vycj, vysj, vxwj, vywj;

    // Jacobian
    std::vector<double> dlefts, drights;

    // Hessian
    Eigen::MatrixXd hess;

    Eigen::IOFormat CommaInitFmt = Eigen::IOFormat(5, Eigen::DontAlignCols, ", ", ", ", "", "", "[ ", " ]");
    Eigen::IOFormat HeavyFmt = Eigen::IOFormat(7, 0, ", ", " ", "\n[", "]", "[", "]");
    Eigen::IOFormat ShortFmt = Eigen::IOFormat(4, 0, ", ", " ", "\n[", "]", "[", "]");

    public:
    Path()
    {
    }

    ~Path()
    {
    }

    public:
    void pose_init(
        const double dt,
        const Eigen::ArrayXd aLefts,
        const Eigen::ArrayXd aRights,
        array3 aStart_pose,
        array3 aStart_twist);

    void pose_init(
        const double dt,
        array3 aStart_pose,
        array3 aStart_twist);

    void pose();

    void loss_init(
        array3 aTarget_pose,
        array3 aTarget_twist,
        array2 aTarget_lr,
        const double aWmax,
        const double aWjerk,
        const double aWback,
        const double ammax
    );

    double losses(bool details=false);
    void jacobian();
    void gradients();
    void seconds();
    void hessian();
    double gradient_descent(int nsteps, double eps);

    double newton_raphson(int nsteps, double eps);
    double newton_raphson_step(double &loss, double &eps);
 
};

#endif // LRPATH_H