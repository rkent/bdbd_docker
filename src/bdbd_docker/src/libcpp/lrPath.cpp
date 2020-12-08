// Given left/right values and constraints, tweak left/right to optimize

#include "lrPath.h"
using namespace std;
using namespace Eigen;

void Path::pose_init(
    const double adt,
    const ArrayXd aLefts,
    const ArrayXd aRights,
    array3 aStart_pose,
    array3 aStart_twist)
{
    lefts = aLefts;
    rights = aRights;
    pose_init(adt, aStart_pose, aStart_twist);
}

void Path::pose_init(
    const double adt,
    array3 aStart_pose,
    array3 aStart_twist)
{
    dt = adt;
    bhxl = bxl * dt;
    bhxr = bxr * dt;
    bhyl = byl * dt;
    bhyr = byr * dt;
    bhol = bol * dt;
    bhor = bor * dt;

    alphax = 1.0 - qx * dt;
    alphay = 1.0 - qy * dt;
    alphao = 1.0 - qo * dt;

    start_pose = aStart_pose;
    start_twist = aStart_twist;
    n = lefts.size();
    
    alphaxj = ArrayXd(n);
    alphayj = ArrayXd(n);
    alphaoj = ArrayXd(n);
    betaj = ArrayXd(n);
    alphaxj(0) = 1.0;
    alphayj(0) = 1.0;
    alphaoj(0) = 1.0;
    betaj(0) = dt;
    for (int i = 1; i < n; i++) {
        alphaxj(i) = alphaxj(i-1) * alphax;
        alphayj(i) = alphayj(i-1) * alphay;
        alphaoj(i) = alphaoj(i-1) * alphao;
        betaj(i) = betaj(i-1) + dt * alphaoj(i);
    }
}

void Path::pose() 
{
    const double px0 = start_pose[0];
    const double py0 = start_pose[1];
    const double theta0 = start_pose[2];
    const double vxw0 = start_twist[0];
    const double vyw0 = start_twist[1];
    const double omega0 = start_twist[2];

    const double vx0 = vxw0 * cos(theta0) + vyw0 * cos(theta0);
    const double vy0 = -vxw0 * sin(theta0) + vyw0 * cos(theta0);

    // twists
    vxj = ArrayXd(n);
    vyj = ArrayXd(n);
    omegaj = ArrayXd(n);

    auto bmotorxj = bhxl * lefts + bhxr * rights;
    auto bmotoryj = bhyl * lefts + bhyr * rights;
    auto bmotoroj = bhol * lefts + bhor * rights;

    for (int i = 1; i < n; i++) {
        // TODO: rewrite these without using dot product
        /*
        if (i == 1) {
            cout << "alphaxj.reverse().tail(i) " << alphaxj.reverse().tail(i) << "\n";
            cout << "bmotorxj.segment(1, i) " << bmotorxj.segment(1, i) << "\n";
            cout << "alphaxj.reverse().tail(i) * bmotorxj.segment(1, i)" << alphaxj.reverse().tail(i) * bmotorxj.segment(1, i) << "\n";
        }
        */
        vxj(i) = vx0 * alphaxj(i) + (alphaxj.reverse().tail(i) * bmotorxj.segment(1, i)).sum();
        vyj(i) = vy0 * alphayj(i) + (alphayj.reverse().tail(i) * bmotoryj.segment(1, i)).sum();
        omegaj(i) = omega0 * alphaoj(i) + (alphaoj.reverse().tail(i) * bmotoroj.segment(1, i)).sum();
    }

    // poses
    pxj = ArrayXd(n);
    pyj = ArrayXd(n);
    thetaj = ArrayXd(n);

    thetaj(0) = theta0;
    for (int i = 1; i < n; i++) {
        thetaj(i) = thetaj(i-1) + dt * omegaj(i);
    }

    cosj = cos(thetaj);
    sinj = sin(thetaj);
    vxcj = vxj * cosj;
    vxsj = vxj * sinj;
    vycj = vyj * cosj;
    vysj = vyj * sinj;
    vxwj = vxcj - vysj;
    vywj = vxsj + vycj;

    pxj(0) = px0;
    pyj(0) = py0;

    for (int i = 1; i < n; i++) {
        pxj(i) = pxj(i-1) + dt * vxwj(i);
        pyj(i) = pyj(i-1) + dt * vywj(i);
        /*
        cout << "i, px, py, vx, vy "
             << i << ' ' 
             << pxj(i) << ' ' 
             << pyj(i) << ' '
             << vxwj(i) << ' '
             << vywj(i) << '\n';
        */
    }
}

void Path::loss_init(
    array3 aTarget_pose,
    array3 aTarget_twist,
    array2 aTarget_lr,
    const double aWmax,
    const double aWjerk,
    const double aWback,
    const double ammax
)
{
    target_pose = aTarget_pose;
    target_twist = aTarget_twist;
    target_lr = aTarget_lr;
    Wmax = aWmax;
    Wjerk = aWjerk;
    Wback = aWback;
    mmax = ammax;
}

double Path::losses(bool details)
{
    // values requiring summing over i
    auto sumMax = 0.1 * Wmax * (
        lefts.pow(10.0).sum() + rights.pow(10.0).sum()
    ) / pow(mmax, 10.0);

    //# backing term
    auto sumBack = 0.1 * Wback * (lefts + rights).min(0.0).pow(10.0).sum();
    // cout << (lefts + rights).min(0.0) << '\n';
    // sumBack = 0.1 * Wback * np.power((lefts + rights).clip(max=0.0), 10).sum()

    double sumJerk = 0.0;
    for (int i = 1; i < n; i++) {
        sumJerk += pow(lefts[i] - lefts[i-1], 2) + pow(rights[i] - rights[i-1], 2);
    }
    sumJerk *= 0.5 * Wjerk; 

    // values based on final targets
    auto last = n - 1;
    const double vals[8] = {
        pxj[last], pyj[last], thetaj[last],
        vxj[last], vyj[last], omegaj[last],
        lefts[last], rights[last]
    };
    const double targets[8] = {
        target_pose[0], target_pose[1], target_pose[2],
        target_twist[0], target_twist[1], target_twist[2],
        target_lr[0], target_lr[1]
    };
    double sumTargets = 0.0;
    for (int i = 0; i < 8; i++) {
        auto diff = vals[i] - targets[i];
        if (i == 2) {
            // this is theta, map to -pi to *pi
            diff = fmod(diff + 3*M_PI, 2*M_PI) - M_PI;
        }
        sumTargets += pow(diff, 2);
    }
    sumTargets *= 0.5;

    lossValue = sumMax + sumJerk + sumTargets + sumBack;
    if (details) {
        cout << "loss " << lossValue << " sumMax " << sumMax << " sumJerk " << sumJerk << " sumTargets " << sumTargets << " sumBack " << sumBack << '\n';
    }
    return lossValue;
}

void Path::jacobian()
{
    auto pxt = target_pose[0];
    auto pyt = target_pose[1];
    auto thetat = target_pose[2];
    auto vxt = target_twist[0];
    auto vyt = target_twist[1];
    auto omegat = target_twist[2];
    auto leftt = target_lr[0];
    auto rightt = target_lr[1];

    auto leftsp9 = (lefts / mmax).pow(9.0);
    auto rightsp9 = (rights / mmax).pow(9.0);
    auto lprsp9 = (lefts + rights).min(0.0).pow(9.0);

    dlefts = {0.0};
    drights = {0.0};
    auto dtheta= fmod(thetaj[n-1] - thetat + 3*M_PI, 2*M_PI) - M_PI;
    for (int k = 1; k < n; k++) {
        dlefts.push_back(
            +(vxj[n-1] - vxt) * bhxl * alphaxj[n-1-k]
            +(vyj[n-1] - vyt) * bhyl * alphayj[n-1-k]
            +(omegaj[n-1] - omegat) * bhol * alphaoj[n-1-k]
            +(dtheta) * bhol * betaj[n-1-k]
            +(pxj[n-1] - pxt) * dpxdl(n-1, k)
            +(pyj[n-1] - pyt) * dpydl(n-1, k)
            +Wmax * leftsp9[k] / mmax
            +Wback * lprsp9[k]
            +Wjerk * (2 * lefts[k] - lefts[k-1] - lefts[min(k+1, n-1)])
        );
        drights.push_back(
            +(vxj[n-1] - vxt) * bhxr * alphaxj[n-1-k]
            +(vyj[n-1] - vyt) * bhyr * alphayj[n-1-k]
            +(omegaj[n-1] - omegat) * bhor * alphaoj[n-1-k]
            +(dtheta) * bhor * betaj[n-1-k]
            +(pxj[n-1] - pxt) * dpxdr(n-1, k)
            +(pyj[n-1] - pyt) * dpydr(n-1, k)
            +Wmax * rightsp9[k]
            +Wback * lprsp9[k]
            +Wjerk * (2 * rights[k] - rights[k-1] - rights[min(k+1, n-1)])
        );
    }
    dlefts[n-1] += (lefts[n-1] - leftt);
    drights[n-1] += (rights[n-1] - rightt);

    return;
}

void Path::gradients()
{
    // gradients
    dpxdl.setZero(n, n);
    dpxdr.setZero(n, n);
    dpydl.setZero(n, n);
    dpydr.setZero(n, n);

    for (int i = 1; i < n; i++) {
        for (int k = 1; k < i + 1; k++) {
            double dotx, doty, doto;
            doto = ((-vxsj.segment(k, i + 1 - k) - vycj.segment(k, i + 1 - k)) * betaj.head(i + 1 -k)).sum();
            dotx = (cosj.segment(k, i + 1 - k) * alphaxj.head(i + 1 -k)).sum();
            doty = (-sinj.segment(k, i + 1 - k) * alphayj.head(i + 1 -k)).sum();
            dpxdl(i, k) = dt * (bhol * doto + bhxl * dotx + bhyl * doty);
            dpxdr(i, k) = dt * (bhor * doto + bhxr * dotx + bhyr * doty);
            // if (i == 1 && k == 1) {
            //     cout << "bhor " << bhor << " doto " << doto << " bhxr " << bhxr;
            //     cout << " dotx " << dotx << " bhyr " << bhyr << " doty " << doty << '\n';
            // }

            doto = ((vxcj.segment(k, i + 1 - k) - vysj.segment(k, i + 1 - k)) * betaj.head(i + 1 -k)).sum();
            dotx = (sinj.segment(k, i + 1 - k) * alphaxj.head(i + 1 -k)).sum();
            doty = (cosj.segment(k, i + 1 - k) * alphayj.head(i + 1 -k)).sum();
            dpydl(i, k) = dt * (bhol * doto + bhxl * dotx + bhyl * doty);
            dpydr(i, k) = dt * (bhor * doto + bhxr * dotx + bhyr * doty);

        }
    }
}

void Path::seconds()
{
    d2pxdldl = ArrayXXd::Zero(n, n);
    d2pxdldr = ArrayXXd::Zero(n, n);
    d2pxdrdr = ArrayXXd::Zero(n, n);
    d2pydldl = ArrayXXd::Zero(n, n);
    d2pydldr = ArrayXXd::Zero(n, n);
    d2pydrdr = ArrayXXd::Zero(n, n);

    for (int j = 1; j < n; j++) {
        // cout << "j " << j << '\n';
        double vxwdt = vxwj[j] * dt;
        double vywdt = vywj[j] * dt;
        double sdt = sinj[j] * dt;
        double cdt = cosj[j] * dt;

        for (int k = 1; k <= j; k++) {
            // cout << "k " << k << '\n';
            double betaljk = betaj[j-k] * bhol;
            double betarjk = betaj[j-k] * bhor;
            double alphaxljk = alphaxj[j-k] * bhxl;
            double alphaxrjk = alphaxj[j-k] * bhxr;
            double alphayljk = alphayj[j-k] * bhyl;
            double alphayrjk = alphayj[j-k] * bhyr;
            for (int m = 1; m <= j; m++) {
                // cout << "m " << m << " j-m " << j-m << "\n";
                double betaljm = betaj[j-m] * bhol;
                double betarjm = betaj[j-m] * bhor;
                double alphaxljm = alphaxj[j-m] * bhxl;
                double alphaxrjm = alphaxj[j-m] * bhxr;
                double alphayljm = alphaxj[j-m] * bhyl;
                double alphayrjm = alphaxj[j-m] * bhyr;

                double sumxll = (
                    -vxwdt * betaljk * betaljm
                    +sdt * (-betaljk * alphaxljm -alphaxljk * betaljm)
                    +cdt * (-betaljk * alphayljm -alphayljk * betaljm)
                );
                double sumxlr = (
                    -vxwdt * betaljk * betarjm
                    +sdt * (-betaljk * alphaxrjm -alphaxljk * betarjm)
                    +cdt * (-betaljk * alphayrjm -alphayljk * betarjm)
                );
                double sumxrr = (
                    -vxwdt * betarjk * betarjm
                    +sdt * (-betarjk * alphaxrjm -alphaxrjk * betarjm)
                    +cdt * (-betarjk * alphayrjm -alphayrjk * betarjm)
                );
                double sumyll = (
                    -vywdt * betaljk * betaljm
                    +sdt * (-betaljk * alphayljm -alphayljk * betaljm)
                    +cdt * (betaljk * alphayljm +alphayljk * betaljm)
                );
                double sumylr = (
                    -vywdt * betaljk * betarjm
                    +sdt * (-betaljk * alphayrjm -alphayljk * betarjm)
                    +cdt * (betaljk * alphayrjm +alphayljk * betarjm)
                );
                double sumyrr = (
                    -vywdt * betarjk * betarjm
                    +sdt * (-betarjk * alphayrjm -alphayrjk * betarjm)
                    +cdt * (betarjk * alphayrjm +alphayrjk * betarjm)
                );

                d2pxdldl(k, m) += sumxll;
                d2pxdldr(k, m) += sumxlr;
                d2pxdrdr(k, m) += sumxrr;
                d2pydldl(k, m) += sumyll;
                d2pydldr(k, m) += sumylr;
                d2pydrdr(k, m) += sumyrr;
            }
        }
    }

}

void Path::hessian()
{
    // second derivative of loss relative to left, rights
    auto pxt = target_pose[0];
    auto pyt = target_pose[1];
    int nh = n - 1;
    // We'll define this as 0 -> nh-1 are lefts[1:], nh -> 2nh-1 are rights[1:]
    hess = MatrixXd(2*nh, 2*nh);

    // values that vary with each k, m value
    auto deltapxn = pxj[nh] - pxt;
    auto deltapyn = pyj[nh] - pyt;
    for (int i = 0; i < 2*nh; i++) {
        auto k = i % nh + 1;

        bool kleft = (i < nh);
        double dpxdu, dpydu, dvxdu, dvydu, domdu, dthdu;
        if (kleft) {
            dpxdu = dpxdl(nh, k);
            dpydu = dpydl(nh, k);
            dvxdu = alphaxj[nh-k] * bhxl;
            dvydu = alphayj[nh-k] * bhyl;
            domdu = alphaoj[nh-k] * bhol;
            dthdu = betaj[nh-k] * bhol;
        }
        else {
            dpxdu = dpxdr(nh, k);
            dpydu = dpydr(nh, k);
            dvxdu = alphaxj[nh-k] * bhxr;
            dvydu = alphayj[nh-k] * bhyr;
            domdu = alphaoj[nh-k] * bhor;
            dthdu = betaj[nh-k] * bhor;
        }

        for (auto j = 0; j < 2 * nh; j++) {
            auto m = j % nh + 1;
            bool mleft = (j < nh);
            double dpxds, dpyds, dvxds, dvyds, domds, dthds, d2px, d2py;
            if (mleft) {
                dpxds = dpxdl(nh, m);
                dpyds = dpydl(nh, m);
                dvxds = alphaxj[nh-m] * bhxl;
                dvyds = alphayj[nh-m] * bhyl;
                domds = alphaoj[nh-m] * bhol;
                dthds = betaj[nh-m] * bhol;
                
                if (kleft) {
                    d2px = d2pxdldl(k, m);
                    d2py = d2pydldl(k, m);
                } else {
                    // note d2pxdrdl[i,j] = d2pxdldr[j,i]
                    d2px = d2pxdldr(m, k);
                    d2py = d2pydldr(m, k);
                }
            }
            else {
                dpxds = dpxdr(nh, m);
                dpyds = dpydr(nh, m);
                dvxds = alphaxj[nh-m] * bhxr;
                dvyds = alphayj[nh-m] * bhyr;
                domds = alphaoj[nh-m] * bhor;
                dthds = betaj[nh-m] * bhor;
                if (kleft) {
                    d2px = d2pxdldr(k, m);
                    d2py = d2pydldr(k, m);
                } else {
                    d2px = d2pxdrdr(k, m);
                    d2py = d2pydrdr(k, m);
                }
            }
            hess(i, j) = (
                deltapxn * d2px + dpxdu * dpxds +
                deltapyn * d2py + dpydu * dpyds +
                dvxdu * dvxds + dvydu * dvyds + domdu * domds + dthdu * dthds
            );
        }
    }
    // values that require k == m
    for (int i = 0; i < 2 * nh; i++) {
        auto k = i % nh + 1;
        bool kleft = (i < nh);
        // max term
        double term = hess(i, i) + 9. * (Wmax / pow(mmax, 2))
            * pow(kleft ? lefts[k] : rights[k], 8);
        // back term
        if (lefts[k] + rights[k] < 0.0 ) {
            term += 9. * Wback * pow(lefts[k] + rights[k], 8);
        }
        // motor target value
        if (k == nh) {
            term += 1.0;
        }
        // jerk term
        term += 2 * Wjerk;
        if (k > 1) {
            hess(i, i-1) -= Wjerk;
        }
        if (k == nh) {
            term -= Wjerk;
        } else {
            hess(i, i+1) -= Wjerk;
        }
        hess(i, i) = term;
    }
}

double Path::gradient_descent(int nsteps, double eps)
{
    double loss = 0.0;
    // perform gradient descent
    for (int count = 0; count < nsteps; count++) {
        pose();
        loss = losses();
        // cout << "gradient_descent loss " << loss << '\n';
        // don't update the last time
        if (count == nsteps - 1) {
            break;
        }
        if (eps == 0.0) {
            return loss;
        }
        gradients();
        jacobian();
        Map<ArrayXd> adlefts(dlefts.data(), dlefts.size());
        Map<ArrayXd> adrights(drights.data(), drights.size());
        auto base_lefts = lefts;
        auto base_rights = rights;

        // limit slew rate
        double maxSlew = 2.0;
        double worst_eps = -1.0;
        int maxi;
        abs(adlefts).maxCoeff(&maxi);
        auto slew_left = abs(adlefts[maxi]);
        abs(adrights).maxCoeff(&maxi);
        auto slew_right = abs(adrights[maxi]);
        auto slew = slew_left > slew_right ? slew_left : slew_right;
        if (slew > maxSlew) {
            worst_eps = maxSlew / slew;
            eps = min(worst_eps, eps);
        }

        auto best_eps = 0.0;
        auto best_loss = loss;
        for (int lcount = 0; lcount < 8; lcount++) {
            if (lcount >= 4 and best_eps > 0.0) {
                break;
            }
            auto last_eps = eps;
            // update left, rights from jacobian
            lefts = base_lefts - eps * adlefts;
            rights = base_rights - eps * adrights;
            pose();
            auto loss = losses();
            if (loss > best_loss) {
                worst_eps = eps;
            } else {
                best_eps = eps;
                best_loss = loss;
            }
            eps = worst_eps < 0.0 ? eps *= 2 : 0.5 * (best_eps + worst_eps);
            // cout << "lcount " << lcount << " loss " << loss << " eps: " << last_eps << " best_eps: " << best_eps << " worst_eps " << worst_eps << '\n';
        }
        eps = best_eps;
        lefts = base_lefts - eps * adlefts;
        rights = base_rights - eps * adrights;
    }
    return loss;
}

double Path::newton_raphson(int nsteps, double eps) {
    double loss = 0.0;
    for (int count = 0; count < nsteps and eps != 0.0; count++) {
        loss = newton_raphson_step(loss, eps);
    }
    loss = losses(true);
    return loss;
}

double Path::newton_raphson_step(double &loss, double &eps)
{
    // use loss == 0.0 first time.
    if (loss == 0.0) {
        pose();
        loss = losses(true);
    }

    auto nh = n-1;
    using dseconds = std::chrono::duration<double>;
    vector<double> times;
    auto start = chrono::steady_clock::now();
    times.push_back(0.0);
    gradients();
    jacobian();
    seconds();
    hessian();
    
    VectorXd mjacobian(2*nh);
    // assemble the full jacobian from the lefts and rights
    for (int i = 0; i < 2*nh; i++) {
        auto k = i % nh + 1;
        bool kleft = (i < nh);
        mjacobian(i) = kleft ? -dlefts[k] : -drights[k];
    }
    VectorXd deltax = hess.partialPivLu().solve(mjacobian);
    times.push_back(dseconds(chrono::steady_clock::now() - start).count());

    //cout << "times ";
    //for (int i = 1; i < times.size()-1; ++i) {
    //    cout << times[i] - times[i-1] << ' ';
    //}
    //cout << "total time: " << times[times.size() - 1] << '\n';

    // line search over deltax looking for best eps
    auto base_lefts = lefts;
    auto base_rights = rights;

    double best_eps = 0.0;
    double best_loss = loss;
    double last_eps = 0.0;
    const double maxSlew = 2.0;

    int maxi;
    abs(deltax.array()).maxCoeff(&maxi);
    auto slew = abs(deltax[maxi]);

    double worst_eps = -1.0;
    if (slew > maxSlew) {
        // cout << "Limiting slew rate\n";
        worst_eps = min(maxSlew / slew, 1.0);
        eps = min(worst_eps, eps);
    }
    bool allDone = false;
    for (int lcount = 0; ; lcount++) {
        // cout << " loss " << loss << " eps: " << last_eps << " best_eps: " << best_eps << " worst_eps " << worst_eps << '\n';
        if ((lcount >= 12 ) or (lcount >= 4 and best_eps > 0.0)) {
            // last time
            allDone = true;
            eps = best_eps;
        }
        // update lefts, rights
        for (int i = 0; i < 2*nh; i++) {
            auto k = i % nh + 1;
            bool kleft = (i < nh);
            if (kleft) {
                lefts[k] = base_lefts[k] + eps * deltax(i);
            } else {
                rights[k] = base_rights[k] + eps * deltax(i);
            }
        }

        pose();
        if (allDone) {
            break;
        }
        loss = losses();

        last_eps = eps;
        if (loss > best_loss) {
            worst_eps = eps;
        } else {
            best_eps = eps;
            best_loss = loss;
        }
        if (worst_eps < 0.0) {
            eps *= 2;
        } else {
            eps = min(1.5, 0.5 * (best_eps + worst_eps));
        }
    }
    return best_loss;
}

