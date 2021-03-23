# use for async handling in ROS with python 3
import rospy
import asyncio

async def asyncMessageSingle(topic, type, timeout=None):
    return rospy.wait_for_message(topic, type, timeout)

def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    print(f"Uncaught async exception: {msg}")

def runner(async_main, final=lambda :print('async runner finished for node:', rospy.get_name())):
    loop = asyncio.get_event_loop()
    # not sure if this is needed or not
    loop.set_exception_handler(handle_exception)

    try:
        loop.run_until_complete(async_main())
    except Exception as exc:
        print('async runner for node {} got exception "{}"'.format(rospy.get_name(), exc))
    finally:
        if final:
            final()
