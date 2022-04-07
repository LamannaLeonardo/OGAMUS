(define (problem ithor-appleinbox)
(:domain ithor)
(:objects
stoveburner_0 - stoveburner
stoveburner_1 - stoveburner
pot_0 - pot
pot_1 - pot
pot_2 - pot
pot_3 - pot
countertop_0 - countertop
countertop_1 - countertop
countertop_2 - countertop
countertop_3 - countertop
countertop_4 - countertop
countertop_5 - countertop
countertop_6 - countertop
countertop_7 - countertop
fridge_0 - fridge
fridge_1 - fridge
statue_0 - statue
coffeemachine_0 - coffeemachine
coffeemachine_1 - coffeemachine
potato_0 - potato
potato_1 - potato
potato_2 - potato
houseplant_0 - houseplant
bowl_0 - bowl
bowl_1 - bowl
cup_0 - cup
diningtable_0 - diningtable
garbagecan_0 - garbagecan
garbagecan_1 - garbagecan
kettle_0 - kettle
pan_0 - pan
spatula_0 - spatula
spatula_1 - spatula
mug_0 - mug
window_0 - window
window_1 - window
bread_0 - bread
soapbottle_0 - soapbottle
soapbottle_1 - soapbottle
apple_0 - apple
drawer_0 - drawer
drawer_1 - drawer
)
(:init
(close_to bowl_0)
(close_to bowl_0)
(close_to bowl_1)
(close_to bowl_1)
(close_to coffeemachine_0)
(close_to coffeemachine_0)
(close_to coffeemachine_1)
(close_to coffeemachine_1)
(close_to countertop_1)
(close_to countertop_1)
(close_to countertop_2)
(close_to countertop_2)
(close_to countertop_4)
(close_to countertop_4)
(close_to cup_0)
(close_to cup_0)
(close_to diningtable_0)
(close_to diningtable_0)
(close_to fridge_0)
(close_to fridge_0)
(close_to fridge_1)
(close_to fridge_1)
(close_to garbagecan_0)
(close_to garbagecan_0)
(close_to houseplant_0)
(close_to houseplant_0)
(close_to kettle_0)
(close_to kettle_0)
(close_to mug_0)
(close_to mug_0)
(close_to pan_0)
(close_to pan_0)
(close_to pot_1)
(close_to pot_1)
(close_to pot_2)
(close_to pot_2)
(close_to potato_0)
(close_to potato_0)
(close_to potato_2)
(close_to potato_2)
(close_to spatula_0)
(close_to spatula_0)
(close_to statue_0)
(close_to statue_0)
(discovered apple_0)
(discovered bowl_0)
(discovered bowl_1)
(discovered bread_0)
(discovered coffeemachine_0)
(discovered coffeemachine_1)
(discovered countertop_0)
(discovered countertop_1)
(discovered countertop_2)
(discovered countertop_3)
(discovered countertop_4)
(discovered countertop_5)
(discovered countertop_6)
(discovered countertop_7)
(discovered cup_0)
(discovered diningtable_0)
(discovered drawer_0)
(discovered drawer_1)
(discovered fridge_0)
(discovered fridge_1)
(discovered garbagecan_0)
(discovered garbagecan_1)
(discovered houseplant_0)
(discovered kettle_0)
(discovered mug_0)
(discovered pan_0)
(discovered pot_0)
(discovered pot_1)
(discovered pot_2)
(discovered pot_3)
(discovered potato_0)
(discovered potato_1)
(discovered potato_2)
(discovered soapbottle_0)
(discovered soapbottle_1)
(discovered spatula_0)
(discovered spatula_1)
(discovered statue_0)
(discovered stoveburner_0)
(discovered stoveburner_1)
(discovered window_0)
(discovered window_1)
(hand_free )
(inspected bowl_0)
(inspected bowl_1)
(inspected coffeemachine_0)
(inspected countertop_0)
(inspected countertop_1)
(inspected countertop_4)
(inspected cup_0)
(inspected fridge_0)
(inspected fridge_1)
(inspected mug_0)
(inspected mug_0)
(inspected potato_0)
(inspected statue_0)
(inspected stoveburner_1)
(open fridge_0)
(open fridge_1)
(openable drawer_0)
(openable drawer_1)
(openable fridge_0)
(openable fridge_1)
(openable kettle_0)
(pickupable apple_0)
(pickupable bowl_0)
(pickupable bowl_1)
(pickupable bread_0)
(pickupable cup_0)
(pickupable kettle_0)
(pickupable mug_0)
(pickupable pan_0)
(pickupable pot_0)
(pickupable pot_1)
(pickupable pot_2)
(pickupable pot_3)
(pickupable potato_0)
(pickupable potato_1)
(pickupable potato_2)
(pickupable soapbottle_0)
(pickupable soapbottle_1)
(pickupable spatula_0)
(pickupable spatula_1)
(pickupable statue_0)
(receptacle bowl_0)
(receptacle bowl_1)
(receptacle coffeemachine_0)
(receptacle coffeemachine_1)
(receptacle countertop_0)
(receptacle countertop_1)
(receptacle countertop_2)
(receptacle countertop_3)
(receptacle countertop_4)
(receptacle countertop_5)
(receptacle countertop_6)
(receptacle countertop_7)
(receptacle diningtable_0)
(receptacle drawer_0)
(receptacle drawer_1)
(receptacle fridge_0)
(receptacle fridge_1)
(receptacle garbagecan_0)
(receptacle garbagecan_1)
(receptacle mug_0)
(receptacle pan_0)
(receptacle pot_0)
(receptacle pot_1)
(receptacle pot_2)
(receptacle pot_3)
(receptacle stoveburner_0)
(receptacle stoveburner_1)
(viewing apple_0)
(viewing bowl_1)
(viewing bread_0)
(viewing countertop_3)
(viewing countertop_6)
(viewing countertop_7)
(viewing drawer_0)
(viewing drawer_1)
(viewing garbagecan_1)
(viewing mug_0)
(viewing pot_3)
(viewing soapbottle_0)
(viewing soapbottle_1)
(viewing spatula_1)
(viewing window_1)
)
(:goal
(and
(exists (?o1 - cabinet) (and (inspected ?o1) (open ?o1) (manipulated ?o1))))
))