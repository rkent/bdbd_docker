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
#include "bdbd_common/MotorsRaw.h"
#include <cmath>
#include <atomic>
#include <algorithm> // std::max

using namespace std;
using namespace Eigen;

// parameters
const double trouble_loss{1.0}; // this means we have lost convergence
const int MAX_STEPS{1000}; // mostly used in debugging to stop at a certain point
const double corr_radius = 0.10; // used to determine effective distance to a point with a theta error
const double converge_ratio{1.e-4}; // when to stop nr iterations

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
    // TODO: surely there is a much simpler way to do this!

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

    // not used but kept as example
    /*
    void timerCB(const ros::TimerEvent& event)
    {
        cout 
            << "x " << xa << " y " << ya << " theta " << thetaa / D_TO_R
            << " vx " << vxa << " vy " << vya << " omega " << omegaa
            << '\n';
    }
    */

};

class LrTweakAction {

protected:
    using dseconds = std::chrono::duration<double>;
    Odom *odom;
    ros::NodeHandle nh;
    actionlib::SimpleActionServer<bdbd_common::LrOptimizeAction> action_server;
    string action_name;
    ros::Publisher feedsub_pub;
    ros::Publisher motors_pub;

public:
    LrTweakAction(Odom* aodom):
        odom(aodom),
        action_server(nh, "/bdbd/dynamicPath", boost::bind(&LrTweakAction::executeCB, this, _1), false),
        action_name{"dynamicPath"},
        feedsub_pub(nh.advertise<bdbd_common::LrResult>("/bdbd/dynamicPath/feedsub", 1)),
        motors_pub(nh.advertise<bdbd_common::MotorsRaw>("/bdbd/motors/cmd_raw", 1))
    {
        action_server.start();
        ROS_INFO("%s: Started", action_name.c_str());
    }

    ~LrTweakAction(void) {}

    void executeCB(const bdbd_common::LrOptimizeGoalConstPtr &goal)
    {
        // This would typically be run in a separate thread
        ROS_INFO("%s: Executing with rate %i", action_name.c_str(), goal->rate);
        //ros::Timer timer = nh.createTimer(ros::Duration(1.0), &Odom::timerCB, &odom);

        const int n = goal->start_lefts.size();
        double loss = 0.0;
        bool success = true;
        array3 start_pose {0.0, 0.0, 0.0};

        const ArrayXd lefts = ArrayXd::Map(goal->start_lefts.data(), n);
        const ArrayXd rights = ArrayXd::Map(goal->start_rights.data(), n);

        // target is calculated relative to initial pose
        const array3 start_pose_map {odom->xa, odom->ya, odom->thetaa};
        array3 now_pose_map;
        auto target_pose_map = transform2d(goal->target_pose, start_pose_map, zero3); 

        bool first_time = true;
        ros::Rate rate(goal->rate);
        double dt {goal->dt};
        const double callback_dt = 1.0 / double(goal->rate);
        Path path;
        array3 old_pose_map {odom->xa, odom->ya, odom->thetaa};

        // Main loop
        for (int step_count = 0; step_count < MAX_STEPS; step_count++) {
            auto start = chrono::steady_clock::now();

            now_pose_map = {odom->xa, odom->ya, odom->thetaa};
            array3 now_twist_robot {odom->vxa, odom->vya, odom->omegaa};
            auto now_pose_start = transform2d(now_pose_map, zero3, start_pose_map);
            double total_time = dt * (n - 1);

            if (!first_time) {
                // Stretch time if needed to keep motors in bounds
                double max_motor{0.0};
                double progress_dt{0.0};
                for (int i = 1; i < n; i++) {
                    max_motor = max(max_motor, abs(path.lefts[i]));
                    max_motor = max(max_motor, abs(path.rights[i]));
                }
                if (max_motor > goal->mmax) {
                    total_time *= (max_motor / goal->mmax);
                }
                // calculate progress toward the previous goal. We'll find an interpolated minimum
                // using the formula on RKJ 2020-10-21 pp 50

                array3 now_pose_old = transform2d(now_pose_map, zero3, old_pose_map);
                double last_distance_sq{0.0};
                double d0{0.0}, d1{0.0}, d2{0.0};
                // move forward until distance starts increasing
                int i = 0;
                for (; i < n; i++) {
                    double dx2 = pow(now_pose_old[0] - path.pxj[i], 2);
                    double dy2 = pow(now_pose_old[1] - path.pyj[i], 2);
                    // Adjust for theta error using RKJ notebook 2020-12-7
                    double dtheta2 = pow(corr_radius * (1.0 - cos(now_pose_old[2] - path.thetaj[i])), 2);
                    double distance_sq = dx2 + dy2 + dtheta2;
                    d0 = d1;
                    d1 = d2;
                    d2 = distance_sq;
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
                // shrink time to move forward by progress, reducing modeling time. Ensure some forward progress.
                progress_dt = max(callback_dt/2, dt * closest_i);

                total_time -= progress_dt;
                if (total_time < callback_dt) {
                    break;
                }
                double new_dt = total_time / (n - 1);

                // Interpolate left, rights to get update values
                ArrayXd new_lefts(n), new_rights(n);
                double tt{progress_dt};
                for (int new_i = 0; new_i < n; ++new_i) {
                    int old_i = trunc(tt / dt);
                    double left, right;
                    if (old_i > n - 2) {
                        left = path.lefts[n-1];
                        right = path.rights[n-1];
                    } else {
                        left = path.lefts[old_i]
                        + (path.lefts[old_i+1] - path.lefts[old_i]) * (tt / dt - (double)old_i);
                        right = path.rights[old_i]
                        + (path.rights[old_i+1] - path.rights[old_i]) * (tt / dt - (double)old_i);
                    }
                    new_lefts[new_i] = left;
                    new_rights[new_i] = right;
                    tt += new_dt;
                }
                dt = new_dt;
                path.lefts = new_lefts;
                path.rights = new_rights;

                // cout << "max_motor " << max_motor << " dt " << dt << '\n';
                path.pose_init(dt, start_pose, now_twist_robot);

            }
            old_pose_map = now_pose_map;

            array3 target_pose = transform2d(target_pose_map, zero3, now_pose_map);
            path.loss_init(
                target_pose
                , goal->target_twist
                , goal->target_lr
                , goal->Wmax
                , goal->Wjerk
                , goal->Wback
                , goal->mmax);

            double last_loss;
            if (first_time) {
                first_time = false;
                path.pose_init(dt, lefts, rights, start_pose, goal->start_twist);
                if (goal->gaussItersMax > 0) {
                    last_loss = path.gradient_descent(goal->gaussItersMax, 1.0);
                    ROS_INFO("%s: Did gradient_descent iters %i loss %f", action_name.c_str(), goal->gaussItersMax, last_loss);
                }
            } else {
                path.pose();
                last_loss = path.losses(false);
            }

            // Main Newton-Raphson iteration
            double eps = 1.0;
            for (int nrIters = 0; nrIters <= goal->nrItersMax; nrIters++) {
                loss = path.newton_raphson_step(loss, eps);
                auto ratio = abs((last_loss - loss) / loss);
                last_loss = loss;
                // ROS_INFO("%s: nr iteration dt=%f eps=%f loss=%f ratio=%g", action_name.c_str(), dt, eps, loss, ratio);
                // cout << "elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
                if (eps == 0.0 || ratio < converge_ratio) {
                    break;
                }
            }

            // Premature exit
            if (action_server.isPreemptRequested()
                || !ros::ok()
                || isnan(loss)
                || loss > trouble_loss)
            {
                if (isnan(loss) || loss > trouble_loss) {
                    ROS_WARN("%s Excessive loss, path convergence failed", action_name.c_str());
                } else {
                    ROS_INFO("%s: Preempted", action_name.c_str());
                }
                action_server.setPreempted();
                success = false;
                break;
            }

            // Normal end of time interval processing

            // set left, right as average over interval callback_dt
            const double forward_time = callback_dt + 0.01;
            double left_sum{0.0}, right_sum{0.0}, tt{0.0};
            for (int i = 1; i < n && tt < forward_time; ++i, tt += dt) {
                double interval_dt = tt + dt < forward_time ? dt : forward_time - tt;
                left_sum += (path.lefts[i-1] + path.lefts[i]) * interval_dt;
                right_sum += (path.rights[i-1] + path.rights[i]) * interval_dt;
            }
            float new_left = 0.5 * left_sum / forward_time;
            float new_right = 0.5 * right_sum / forward_time;
            bdbd_common::MotorsRaw motors;
            motors.left = new_left;
            motors.right = new_right;
            motors_pub.publish(motors);

            loss = path.losses(false);
            bdbd_common::LrResult feedback;
            feedback.loss = loss;
            feedback.dt = dt;
            feedback.now_pose_map = now_pose_map;
            for (int i = 0; i < n; ++i) {
                feedback.lefts.push_back(path.lefts[i]);
                feedback.rights.push_back(path.rights[i]);
                feedback.pxj.push_back(path.pxj[i]);
                feedback.pyj.push_back(path.pyj[i]);
                feedback.thetaj.push_back(path.thetaj[i]);
                feedback.vxj.push_back(path.vxj[i]);
                feedback.vyj.push_back(path.vyj[i]);
                feedback.omegaj.push_back(path.omegaj[i]);
            }
            feedsub_pub.publish(feedback);

            rate.sleep();
        }

        // End of path
        bdbd_common::MotorsRaw motors;
        motors.left = 0.0;
        motors.right = 0.0;
        motors_pub.publish(motors);
        if (success) {
            ROS_INFO("%s: Succeeded", action_name.c_str());
            bdbd_common::LrOptimizeResult result;
            result.loss = loss;
            result.now_pose_map = now_pose_map;
            result.dt = dt;
            for (int i = 0; i < n; ++i) {
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
    ros::NodeHandle nh;
    ROS_INFO("dynamicPath starting up");

    Odom odom;
    ros::Subscriber odometrySub = nh.subscribe("/t265/odom/sample", 10, &Odom::odometryCB, &odom);

    LrTweakAction lrTweakAction(&odom);
    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    return(0);
};
