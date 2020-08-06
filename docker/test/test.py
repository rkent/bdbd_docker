# testing docker logs

import time
count = 0
while True:
    print('Now {}'.format(time.time()), flush=True)
    time.sleep(1)
    count += 1
    if count > 10:
        raise "This is an exception"
