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

double amean(const double a1, const double a2, const double f) {
    // https://en.wikipedia.org/wiki/Mean_of_circular_quantities
    double sina = (1. - f) * sin(a1) + f * sin(a2);
    double cosa = (1. - f) * cos(a1) + f * cos(a2);
    return atan2(sina, cosa);
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

        array3 start_pose = {0.0, 0.0, 0.0};

        const ArrayXd lefts = ArrayXd::Map(goal->start_lefts.data(), n);
        const ArrayXd rights = ArrayXd::Map(goal->start_rights.data(), n);

        // target is calculated relative to initial pose
        array3 target_pose_map;
        target_pose_map[0] = odom->xa + goal->target_pose[0] - goal->start_pose[0];
        target_pose_map[1] = odom->ya + goal->target_pose[1] - goal->start_pose[1];
        target_pose_map[2] = odom->thetaa + goal->target_pose[2] - goal->start_pose[2];

        cout << "init time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
        // main loop
        bool allok = true;
        bool first_time = true;
        ros::Rate rate(20);
        double dt {goal->dt};
        Path path;
        while (true) {
            auto start = chrono::steady_clock::now();
    
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

            if (first_time && goal->gaussItersMax > 0) {
                path.pose_init(dt, lefts, rights, start_pose, goal->start_twist);
                path.pose();

                //ROS_INFO("%s: Doing gradient_descent iters %i", action_name.c_str(), goal->gaussItersMax);
                first_time = false;
                loss = path.gradient_descent(goal->gaussItersMax, 1.0);
                loss = path.losses(false);
                // cout << "gauss loss " << loss << " elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
            } else {
                // Stretch dt if needed to keep motors in bounds
                double max_motor = 0.0;
                for (int i = 1; i < lefts.size(); i++) {
                    max_motor = max(max_motor, abs(path.lefts[i]));
                    max_motor = max(max_motor, abs(path.rights[i]));
                }
                if (max_motor > goal->mmax) {
                    dt *= (max_motor / goal->mmax);
                }
                cout << "max_motor " << max_motor << " dt " << dt << '\n';
                path.pose_init(dt, start_pose, goal->start_twist);
                path.pose();
            }

            double eps = 1.0;
            auto last_loss = loss;
            for (int nrIters = 1; nrIters <= goal->nrItersMax; nrIters++) {
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
                // ROS_INFO("%s: nr iteration eps=%f loss=%f conv_ratio=%g", action_name.c_str(), eps, loss, ratio);
                // cout << "elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
                if (eps == 0.0 || ratio < converge_ratio) {
                    break;
                }
            }
            if (allok) {
                loss = path.losses(false);
                // TODO: Insert test for done
                bdbd_common::LrResult feedback;
                feedback.loss = loss;
                for (int i = 0; i < path.lefts.size(); ++i) {
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
            /*
            bdbd_common::LrOptimizeResult result;
            result.loss = loss;
            for (int i = 0; i < path.lefts.size(); ++i) {
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
            cout << "C++ action elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
            */
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
    ros::Timer timer = nh.createTimer(ros::Duration(1.0), &Odom::timerCB, &odom);
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return(0);
};
