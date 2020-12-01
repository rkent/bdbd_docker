// Given left/right values and constraints, tweak left/right to optimize

#include "lrPath.cpp"

class LrTweakAction {
    using dseconds = std::chrono::duration<double>;
    chrono::steady_clock::time_point start;

protected:
    ros::NodeHandle nh;
    actionlib::SimpleActionServer<bdbd_common::LrOptimizeAction> action_server;
    string action_name;

public:
    LrTweakAction(string name):
        action_server(nh, "/bdbd/lrTweak", boost::bind(&LrTweakAction::executeCB, this, _1), false),
        action_name(name)
    {
        action_server.start();
        ROS_INFO("%s: Started", action_name.c_str());
    }

    ~LrTweakAction(void) {}

    void executeCB(const bdbd_common::LrOptimizeGoalConstPtr &goal)
    {
        auto n = goal->start_lefts.size();
        double loss = 0.0;
        bool success = true;
        start = chrono::steady_clock::now();

        Path path(goal->dt);
        const ArrayXd lefts = ArrayXd::Map(goal->start_lefts.data(), n);
        const ArrayXd rights = ArrayXd::Map(goal->start_rights.data(), n);
        path.pose_init(lefts, rights, goal->start_pose, goal->start_twist);
        path.loss_init(
            goal->target_pose
            , goal->target_twist
            , goal->target_lr
            , goal->Wmax
            , goal->Wjerk
            , goal->Wback
            , goal->mmax);

        ROS_INFO("%s: Executing", action_name.c_str());
        if (goal->gaussItersMax > 0) {
            ROS_INFO("%s: Doing gradient_descent iters %i", action_name.c_str(), goal->gaussItersMax);
            loss = path.gradient_descent(goal->gaussItersMax, 1.0);
            loss = path.losses(true);
            cout << "gauss loss " << loss << '\n';
        }

        double eps = 1.0;
        for (int nrIters = 1; nrIters <= goal->nrItersMax; nrIters++) {
            if (action_server.isPreemptRequested() || !ros::ok()) {
                ROS_INFO("%s: Preempted", action_name.c_str());
                action_server.setPreempted();
                success = false;
                break;
            }
            loss = path.newton_raphson_step(loss, eps);
            ROS_INFO("%s: nr iteration eps=%f loss=%f", action_name.c_str(), eps, loss);
            if (eps == 0.0) {
                break;
            }
        }
        if (success) {
            ROS_INFO("%s: Succeeded", action_name.c_str());
            // print out loss details
            loss = path.losses(true);
            bdbd_common::LrOptimizeResult result;
            result.loss = loss;
            for (int i = 0; i < path.lefts.size(); ++i) {
                result.lefts.push_back(path.lefts[i]);
                result.rights.push_back(path.rights[i]);
                result.pxs.push_back(path.pxj[i]);
                result.pys.push_back(path.pyj[i]);
                result.thetas.push_back(path.thetaj[i]);
                result.vxs.push_back(path.vxj[i]);
                result.vys.push_back(path.vyj[i]);
                result.omegas.push_back(path.omegaj[i]);
            }
            action_server.setSucceeded(result);
            cout << "C++ action elapsed time " << dseconds(chrono::steady_clock::now() - start).count() << '\n';
        }
    }

};

void rawLRcallback(const bdbd_common::LeftRightsConstPtr& leftRights)
{
    cout << "\n***** received LR *****\n";
    using dseconds = std::chrono::duration<double>;
    auto start = chrono::steady_clock::now();

    auto dt = leftRights->dt;
    auto msgLefts = leftRights->lefts;
    auto msgRights = leftRights->rights;
    auto n = leftRights->lefts.size();

    const array3 start_pose {0.0, 0.0, 0.0};
    const array3 start_twist {0.0, 0.0, 0.0};
    Path path(dt);
 
    const ArrayXd lefts = ArrayXd::Map(msgLefts.data(), n);
    const ArrayXd rights = ArrayXd::Map(msgRights.data(), n);

    path.pose_init(lefts, rights, start_pose, start_twist);
    const array3 target_pose = {0.2, 0.1, 180 * D_TO_R};
    const array3 target_twist = {0.0, 0.0, 0.0};
    const array2 target_lr = {0.0, 0.0};
    const double Wmax = dt * 5.e-4;
    const double Wjerk = dt * 1.e-3;
    const double Wback = 1.0;
    const double mmax = 1.0;

    path.loss_init(target_pose, target_twist, target_lr, Wmax, Wjerk, Wback, mmax);
    // auto loss = path.gradient_descent(10, 0.5);
    // ROS_INFO_STREAM("final loss is " << loss);
    path.newton_raphson(10, 1.0);

    auto nr_time = ((dseconds)(chrono::steady_clock::now() - start)).count();

    double lrMax = 0.0;
    for (auto i = 0; i < path.lefts.size(); i++) {
        if (abs(path.lefts[i]) > lrMax) lrMax = abs(path.lefts[i]);
        if (abs(path.rights[i]) > lrMax) lrMax = abs(path.rights[i]);

    }
    cout << " lrMax " << lrMax;
    cout << " n " << 2*(path.lefts.size() - 1);
    cout << " sim time " << dt * (path.lefts.size() - 1);
    cout << " nr time " << nr_time;
    cout << '\n';

    cout << " lefts" << path.lefts.format(path.CommaInitFmt) << '\n';
    cout << " rights" << path.rights.format(path.CommaInitFmt) << '\n';
    cout << " pxj" << path.pxj.format(path.CommaInitFmt) << '\n';
    cout << " pyj" << path.pyj.format(path.CommaInitFmt) << '\n';
    //path.pose();
    //auto loss = path.losses();
    //path.gradients();
    //path.jacobian();

    //Map<ArrayXXd> adpydr(path.dpydr.data(), n, n);
    //Map<ArrayXd> adlefts(path.dlefts.data(), path.dlefts.size());
    //Map<ArrayXd> adrights(path.drights.data(), path.drights.size());
    // ROS_INFO_STREAM("dpydr: " << adpydr.format(path.HeavyFmt) << '\n');
    // ROS_INFO_STREAM("dlefts: " << adlefts.format(path.CommaInitFmt) <<'\n');
    // ROS_INFO_STREAM("drights: " << adrights.format(path.CommaInitFmt) << '\n');

}

int main(int argc, char **argv)
{
    printf("lrTweak here\n");
    cout.setf(std::ios::unitbuf);
    ros::init(argc, argv, "lrTweak");
    ROS_INFO("lrTweak starting up");
    ros::NodeHandle nh;
    ros::Subscriber lrSub = nh.subscribe("rawLR", 10, rawLRcallback);

    LrTweakAction trTweakAction("lrTweak");
    ros::spin();

    return(0);
};
