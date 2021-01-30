import math
from bdbd_common.utils import fstr, gstr

def speedPlan(lp, v0=0.0, vn=0.0, vc=0.3, u=0.25):
    # plan for speeds in a static path, with maximum speed slew rate
    # lp: total path length
    # v0, vn: start, finish speeds
    # vc: cruise speed
    # u: time allowed to slew from 0 to vc

    print(fstr({'v0': v0, 'vn': vn, 'vc': vc, 'lp': lp, 'u': u}))
    if vc <= 0.:
        raise Exception('vc must be positive, non-zero')
    s_plan = []

    # try ramp to vc, then to vn
    S = None if u == 0.0 else vc / u
    d0 = abs(vc*vc - v0*v0) / (2*S) if S else None
    d2 = abs(vc*vc - vn*vn) / (2*S) if S else None

    if S and d0 + d2 < lp:
        d1 = (lp - d0 - d2)
        t1 = d1 / vc
        s_plan = [
            {
                'start': 0.0,
                'end': d0,
                'vstart': v0,
                'vend': vc,
                'time': (vc - v0) / S
            },
            {
                'start': d0,
                'end': d0 + d1,
                'vstart': vc,
                'vend': vc,
                'time': t1
            },
            {
                'start': d0 + d1,
                'end': lp,
                'vstart': vc,
                'vend': vn,
                'time': (vc - vn) / S
            }
        ]

    else:
        # try to increase to a speed vm, then decelerate to vn
        dp = (v0*v0 + vn*vn) / (2.0 * S)
        vm2 = S * (lp + dp)
        vm = math.sqrt(vm2)

        if vm > v0 and vm > vn:
            # Case 2a: t1 = 0, 2-segment ramp up then down
            d0 = (vm*vm - v0*v0) / (2.0 * S)
            s_plan = [
                {
                    'start': 0.0,
                    'end': d0,
                    'vstart': v0,
                    'vend': vm,
                    'time': (vm - v0) / S
                },
                {
                    'start': d0,
                    'end': lp,
                    'vstart': vm,
                    'vend': vn,
                    'time': (vm - vn) / S
                }
            ]
        else:
            # just ramp from v0 to vn
            s_plan = [
                {
                    'start': 0.0,
                    'end': lp,
                    'vstart': v0,
                    'vend': vn,
                    'time': 2 * lp / (v0 + vn)
                }
            ]

    return s_plan

if __name__ == '__main__':
    lp = 0.01
    while lp < 0.1:
        s_plan = speedPlan(lp, v0=0.30, vc=0.30, vn=0.0, u=0.25)
        print(fstr(s_plan, n_per_line=1, fmat='9.6f'))
        lp += 0.01
