// Given left/right values and constraints, tweak left/right to optimize

#include "libcpp/lrPath.h"

#include <ros/ros.h>
#include <ros/console.h>
#include <actionlib/server/simple_action_server.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include "bdbd_common/LeftRights.h"
#include "bdbd_common/LrOptimizeAction.h"
#include "bdbd_common/LrResult.h"
#include <cmath>
#include <atomic>
#include <algorithm> // std::max

using namespace std;
using namespace Eigen;

// const double D_TO_R = 3.1415926535 / 180.; // defined in lrPath.h
const array3 zero3{0.0, 0.0, 0.0};

double amean(const double a1, const double a2, const double f) {
    // https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    double sina = (1. - f) * sin(a1) + f * sin(a2);
    double cosa = (1. - f) * cos(a1) + f * cos(a2);
    return atan2(sina, cosa);
}

array3 transform2d(const array3 poseA, const array3 frameA, const array3 frameB)
{
    // transform poseA from frameA to frameC.
    // See RKJ 2020-10-1 pp 42-43

    // these are in world frame M
    const double AxM {frameA[0]};
    const double AyM {frameA[1]};
    const double AthetaM {frameA[2]};
    const double BxM {frameB[0]};
    const double ByM {frameB[1]};
    const double BthetaM {frameB[2]};

    // input pose is in frame A
    const double xA {poseA[0]};
    const double yA {poseA[1]};
    const double thetaA {poseA[2]};

    // transform B origin from world frame M to A
    double BxA = (BxM - AxM) * cos(AthetaM) + (ByM - AyM) * sin(AthetaM);
    double ByA = (ByM - AyM) * cos(AthetaM) -(BxM - AxM) * sin(AthetaM);

    // translate point from A-relative to B-relative in A orientation
    double xAB = xA - BxA;
    double yAB = yA - ByA;

    // rotate point to B orientation
    double theta = BthetaM - AthetaM;
    double xB = xAB * cos(theta) + yAB * sin(theta);
    double yB = yAB * cos(theta) - xAB * sin(theta);
    double thetaB = thetaA - theta;
    return array3{xB, yB, thetaB}; 
}

class Odom {
public:
    atomic<double> xa, ya, thetaa, vxa, vya, omegaa;
    double new_factor;
    bool first_call;

    Odom() {
        xa = 0.0;
        ya = 0.0;
        thetaa = 0.0;
        vxa = 0.0;
        vya = 0.0;
        omegaa = 0.0;
        new_factor = 0.1;
        first_call = true;
    }

    void odometryCB(const nav_msgs::OdometryConstPtr& odometry)
    {
        double x = odometry->pose.pose.position.x;
        double y = odometry->pose.pose.position.y;
        double theta = tf::getYaw(odometry->pose.pose.orientation);
        double vx = odometry->twist.twist.linear.x;
        double vy = odometry->twist.twist.linear.y;
        double omega = odometry->twist.twist.angular.z;

        //cout << "***** received odometry *****\n";
        if (first_call) {
            xa = x;
            ya = y;
            thetaa = theta;
            vxa = vx;
            vya = vy;
            omegaa = omega;
            first_call = false;
        } else {
            xa = (1. - new_factor) * xa + new_factor * x;
            ya = (1. - new_factor) * ya + new_factor * y;
            thetaa = amean(thetaa, theta, new_factor);
            vxa = (1. - new_factor) * vxa + new_factor * vx;
            vya = (1. - new_factor) * vya + new_factor * vy;
            omegaa = (1. - new_factor) * omegaa + new_factor * omega;
        }
    }

    void timerCB(const ros::TimerEvent& event)
    {
        cout 
            << "x " << xa << " y " << ya << " theta " << thetaa / D_TO_R
            << " vx " << vxa << " vy " << vya << " omega " << omegaa
            << '\n';
    }

};

class LrTweakAction {
protected:
    using dseconds = std::chrono::duration<double>;
    chrono::steady_clock::time_point start;

    ros::NodeHandle nh;
    actionlib::SimpleActionServer<bdbd_common::LrOptimizeAction> action_server;
    string action_name;
    // starting position in map coordinates
    double pxm_start, pxy_start, thetam_start;
    // convergence parameter
    double converge_ratio;
    Odom *odom;
    ros::Publisher feedback_pub;

public:
    LrTweakAction(string name, Odom* aodom):
        action_server(nh, "/bdbd/lrTweak", boost::bind(&LrTweakAction::executeCB, this, _1), false),
        action_name(name),
        converge_ratio(1.e-4),
        odom(aodom),
        feedback_pub(nh.advertise<bdbd_common::LrResult>("dynamicPath/feedback", 1))
    {
        action_server.start();
        ROS_INFO("%s: Started", action_name.c_str());
    }

    ~LrTweakAction(void) {}

    void executeCB(const bdbd_common::LrOptimizeGoalConstPtr &goal)
    {
        ROS_INFO("%s: Executing", action_name.c_str());
        using dseconds = std::chrono::duration<double>;
        pxm_start = odom->xa;
        pxy_start = odom->ya;
        thetam_start = odom->thetaa;
        auto n = goal->start_lefts.size();
        double loss = 0.0;
        bool success = true;
        start = chrono::steady_clock::now();

        array3 start_pose {0.0, 0.0, 0.0};

        const ArrayXd lefts = ArrayXd::Map(goal->start_lefts.data(), n);
        const ArrayXd rights = ArrayXd::Map(goal->start_rights.data(), n);

        // target is calculated relative to initial pose
        const array3 start_pose_map {odom->xa, odom->ya, odom->thetaa};
        array3 target_pose_map;
        for (int i = 0; i < 3; ++i) {
            target_pose_map[i] = start_pose_map[i]+ goal->target_pose[i] - goal->start_pose[i];
        }

        bool allok = true;
        bool first_time = true;
        ros::Rate rate(goal->rate);
        double dt {goal->dt};
        double callback_dt = 1.0 / double(goal->rate);
        Path path;
        array3 old_pose_map {odom->xa, odom->ya, odom->thetaa};
        double last_total_time{1.e10}, total_time{0.0}, new_total_time{1.e10};

        // Main loop
        while (true) {
            auto start = chrono::steady_clock::now();
            array3 now_pose_map {odom->xa, odom->ya, odom->thetaa};
            
            // calculate progress toward the previous goal. We'll find an interpolated minimum using the formula
            // on RKJ 2020-10-21 pp 50
            double const corr_radius = 0.10;
            if (!first_time) {
                array3 now_pose_old = transform2d(now_pose_map, zero3, old_pose_map);
                double last_distance_sq{0.0};
                double d0{0.0}, d1{0.0}, d2{0.0};
                int i = 0;
                // move forward until distance starts increasing
                for (; i < n; i++) {
                    double dx2 = pow(now_pose_old[0] - path.pxj[i], 2);
                    double dy2 = pow(now_pose_old[1] - path.pyj[i], 2);
                    double dtheta2 = pow(corr_radius * (1.0 - cos(now_pose_old[2] - path.thetaj[i])), 2);
                    double distance_sq = dx2 + dy2 + dtheta2;
                    // Adjust for theta error using RKJ notebook 2020-12-7
                    distance_sq += pow(corr_radius * (1.0 - cos(now_pose_old[2] - path.thetaj[i])), 2);
                    d0 = d1;
                    d1 = d2;
                    d2 = distance_sq;
                    cout << "i " << i << " distance_sq " << distance_sq
                        << " dx2 " << dx2 << " dy2 "<< dy2 << " dtheta2 " << dtheta2 << '\n';
                    if (distance_sq > last_distance_sq && i >= 2) {
                        break;
                    }
                    last_distance_sq = distance_sq;
                }
                double closest_i;
                double denom = 2. * (d0 + d2 - 2*d1);
                if (denom == 0.0) {
                    closest_i = i-1;
                } else {
                    closest_i = (i-2) + (3.*d0 + d2 - 4.*d1) / denom;
                }
                closest_i = min((double)i, max((double)(i-2), closest_i));
                double progress_dt = max(0.0, dt * closest_i);
                // shrink dt to move forward by progress, reducing modeling time
                total_time = dt * (n - 1);
                new_total_time = max(0.25, (total_time - progress_dt));
                // Near the end, bring to a firm end
                if (last_total_time <= 0.25) {
                    new_total_time = last_total_time - callback_dt;
                }
                dt = new_total_time / (n - 1);
                cout << "progress_dt " << progress_dt << " dt " << dt 
                    << " total_time " << total_time << " closest_i " << closest_i <<'\n';
            }
            last_total_time = new_total_time;
            if (new_total_time < callback_dt) {
                break;
            }
            old_pose_map = now_pose_map;
            array3 target_pose;
            target_pose[0] = target_pose_map[0] - odom->xa;
            target_pose[1] = target_pose_map[1] - odom->ya;
            target_pose[2] = target_pose_map[2] - odom->thetaa;
            /*
            for (int i = 0; i < 3; i++) {
                cout << " target_pose[" << i << "] " << target_pose[i];
            }
            cout  << '\n';
            */

            path.loss_init(
                target_pose
                , goal->target_twist
                , goal->target_lr
                , goal->Wmax
                , goal->Wjerk
                , goal->Wback
                , goal->mmax);

            if (first_time) {
                first_time = false;
                path.pose_init(dt, lefts, rights, start_pose, goal->start_twist);
                if (goal->gaussItersMax > 0) {
                    //ROS_INFO("%s: Doing gradient_descent iters %i", action_name.c_str(), goal->gaussItersMax);
                    loss = path.gradient_descent(goal->gaussItersMax, 1.0);
                    cout << "gauss loss " << loss << " elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
                }
            } else {
                // Stretch dt if needed to keep motors in bounds
                double max_motor = 0.0;
                for (int i = 1; i < n; i++) {
                    max_motor = max(max_motor, abs(path.lefts[i]));
                    max_motor = max(max_motor, abs(path.rights[i]));
                }
                if (max_motor > goal->mmax) {
                    dt *= (max_motor / goal->mmax);
                    cout << " dt stretched to " << dt << '\n';
                }
                // cout << "max_motor " << max_motor << " dt " << dt << '\n';
                path.pose_init(dt, start_pose, goal->start_twist);
            }
            path.pose();

            double eps = 1.0;
            auto last_loss = path.losses(false);
            for (int nrIters = 0; nrIters <= goal->nrItersMax; nrIters++) {
                if (action_server.isPreemptRequested() || !ros::ok()) {
                    ROS_INFO("%s: Preempted", action_name.c_str());
                    action_server.setPreempted();
                    success = false;
                    allok = false;
                    break;
                }
                loss = path.newton_raphson_step(loss, eps);
                auto ratio = abs((last_loss - loss) / loss);
                last_loss = loss;
                // ROS_INFO("%s: nr iteration dt=%f eps=%f loss=%f conv_ratio=%g lefts[10]=%f", action_name.c_str(), dt, eps, loss, ratio, path.lefts[10]);
                // cout << "elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
                if (eps == 0.0 || ratio < converge_ratio) {
                    break;
                }
            }

            if (allok) {

                // set left, right as average over interval callback_dt
                double left_sum{0.0}, right_sum{0.0}, tt{0.0};
                for (int i = 1; i < n && tt < callback_dt; ++i, tt += dt) {
                    double interval_dt = tt + dt < callback_dt ? dt : callback_dt - tt;
                    left_sum += (path.lefts[i-1] + path.lefts[i]) * interval_dt;
                    right_sum += (path.rights[i-1] + path.rights[i]) * interval_dt;
                    // cout << "interval_dt " << interval_dt << " left_sum " << left_sum << '\n';
                }
                double new_left = 0.5 * left_sum / callback_dt;
                double new_right = 0.5 * right_sum / callback_dt;
                cout << "new_left " << new_left << " new_right " << new_right << " callback_dt " << callback_dt << '\n';

                loss = path.losses(true);
                // TODO: Insert test for done
                bdbd_common::LrResult feedback;
                feedback.loss = loss;
                for (int i = 0; i < n; ++i) {
                    feedback.dt = dt;
                    feedback.lefts.push_back(path.lefts[i]);
                    feedback.rights.push_back(path.rights[i]);
                    feedback.pxj.push_back(path.pxj[i]);
                    feedback.pyj.push_back(path.pyj[i]);
                    feedback.thetaj.push_back(path.thetaj[i]);
                    feedback.vxj.push_back(path.vxj[i]);
                    feedback.vyj.push_back(path.vyj[i]);
                    feedback.omegaj.push_back(path.omegaj[i]);
                }
                feedback_pub.publish(feedback);
            } else {
                break;
            }

            rate.sleep();
        }
        if (success) {
            ROS_INFO("%s: Succeeded", action_name.c_str());
            bdbd_common::LrOptimizeResult result;
            result.loss = loss;
            for (int i = 0; i < n; ++i) {
                result.dt = dt;
                result.lefts.push_back(path.lefts[i]);
                result.rights.push_back(path.rights[i]);
                result.pxj.push_back(path.pxj[i]);
                result.pyj.push_back(path.pyj[i]);
                result.thetaj.push_back(path.thetaj[i]);
                result.vxj.push_back(path.vxj[i]);
                result.vyj.push_back(path.vyj[i]);
                result.omegaj.push_back(path.omegaj[i]);
            }
            action_server.setSucceeded(result);
        }
    }

};

int main(int argc, char **argv)
{
    cout.setf(std::ios::unitbuf);
    ros::init(argc, argv, "dynamicPath");
    ROS_INFO("dynamicPath starting up");
    ros::NodeHandle nh;
    Odom odom;
    ros::Subscriber odometrySub = nh.subscribe("/t265/odom/sample", 10, &Odom::odometryCB, &odom);

    LrTweakAction lrTweakAction("dynamicPath", &odom);
    //ros::Timer timer = nh.createTimer(ros::Duration(1.0), &Odom::timerCB, &odom);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return(0);
};
