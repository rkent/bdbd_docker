# used to test features

from numpy import linspace
from bdbd_common.utils import sstr
import cmath
import math

# intersection over area in polar coordinates. Theta is in the
# horizontal plane, phi in the vertical plane. 0, 0 is straight ahead.

# a box is represented by (vCenter, width, height) where v is
# the complex vector ( (cos(theta) + j sin(theta)), (cos(phi) + j sin(phi)) )

# for an optical frame, x is to the right, and y is down. 'upper' is less than 'lower'
def intersection_over_union(boxA, boxB):
    
    (vCenterA, widthA, heightA) = boxA
    (vCenterAx, vCenterAy) = vCenterA
    (vCenterB, widthB, heightB) = boxB
    (vCenterBx, vCenterBy) = vCenterB

    # rotations from center)
    rxA = math.cos(widthA/2) + 1j * math.sin(widthA/2)
    ryA = math.cos(heightA/2) + 1j * math.sin(heightA/2)
    rxB = math.cos(widthB/2) + 1j * math.sin(widthB/2)
    ryB = math.cos(heightB/2) + 1j * math.sin(heightB/2)
    print(sstr('rxA ryA rxB ryB'))

    # corners
    (vlrxA, vlryA) = (vCenterAx * rxA, vCenterAy * ryA)
    (vulxA, vulyA) = (vCenterAx * rxA.conjugate(), vCenterAy * ryA.conjugate())
    (vlrxB, vlryB) = (vCenterBx * rxB, vCenterBy * ryB)
    (vulxB, vulyB) = (vCenterBx * rxB.conjugate(), vCenterBy * ryB.conjugate())
    print(sstr('vulxA vulyA vlrxA vlryA vulxB vulyB vlrxB vlryB'))

    # get angles to determine max, min for intersection
    dxmin = vulxB / vulxA
    dxmax = vlrxB / vlrxA
    dymin = vulyB / vulyA
    dymax = vlryB / vlryA 
    print(sstr('dxmin dxmax dymin dymax'))

# convert to angles, using 45 degree FOV at 800 pixels
D_TO_R = math.pi / 180.
cf = 45 * D_TO_R / 800.

# boxes as radians (theta, phi) from upper left hand corner
rboxA = (100 * cf, 200 * cf)
widthA = 30 * cf
heightA = 70 * cf
rboxB = (120 * cf, 260 * cf)
widthB = 40 * cf
heightB = 50 * cf

centerA = (
        math.cos(rboxA[0]) + 1j * math.sin(rboxA[0]),
        math.cos(rboxA[1]) + 1j * math.sin(rboxA[1])
    )

centerB = (
        math.cos(rboxB[0]) + 1j * math.sin(rboxB[0]),
        math.cos(rboxB[1]) + 1j * math.sin(rboxB[1])
    )
boxA = (centerA, widthA, heightA)
boxB = (centerB, widthB, heightB)

print(sstr('boxA boxB'))
rr = intersection_over_union(boxA, boxB)
