(define (problem ithor-appleinbox)
(:domain ithor)
(:objects
showerdoor_0 - showerdoor
showerdoor_1 - showerdoor
painting_0 - painting
towel_0 - towel
towelholder_0 - towelholder
soapbottle_0 - soapbottle
countertop_0 - countertop
countertop_1 - countertop
countertop_2 - countertop
toiletpaperhanger_0 - toiletpaperhanger
spraybottle_0 - spraybottle
garbagecan_0 - garbagecan
toilet_0 - toilet
sink_0 - sink
sink_1 - sink
sink_2 - sink
faucet_0 - faucet
faucet_1 - faucet
mirror_0 - mirror
mirror_1 - mirror
candle_0 - candle
sinkbasin_0 - sinkbasin
sinkbasin_1 - sinkbasin
)
(:init
(close_to countertop_0)
(close_to countertop_0)
(close_to countertop_1)
(close_to countertop_1)
(close_to countertop_2)
(close_to countertop_2)
(close_to faucet_0)
(close_to faucet_0)
(close_to faucet_1)
(close_to faucet_1)
(close_to mirror_0)
(close_to mirror_0)
(close_to mirror_1)
(close_to mirror_1)
(close_to sink_0)
(close_to sink_0)
(close_to sink_1)
(close_to sink_1)
(close_to sink_2)
(close_to sink_2)
(close_to sinkbasin_0)
(close_to sinkbasin_0)
(close_to sinkbasin_1)
(close_to sinkbasin_1)
(close_to soapbottle_0)
(close_to soapbottle_0)
(discovered candle_0)
(discovered countertop_0)
(discovered countertop_1)
(discovered countertop_2)
(discovered faucet_0)
(discovered faucet_1)
(discovered garbagecan_0)
(discovered mirror_0)
(discovered mirror_1)
(discovered painting_0)
(discovered showerdoor_0)
(discovered showerdoor_1)
(discovered sink_0)
(discovered sink_1)
(discovered sink_2)
(discovered sinkbasin_0)
(discovered sinkbasin_1)
(discovered soapbottle_0)
(discovered spraybottle_0)
(discovered toilet_0)
(discovered toiletpaperhanger_0)
(discovered towel_0)
(discovered towelholder_0)
(hand_free )
(inspected countertop_0)
(inspected countertop_1)
(inspected countertop_2)
(inspected countertop_2)
(inspected faucet_0)
(inspected faucet_0)
(inspected faucet_1)
(inspected faucet_1)
(inspected mirror_0)
(inspected mirror_1)
(inspected mirror_1)
(inspected sink_0)
(inspected sink_1)
(inspected sink_1)
(inspected sink_2)
(inspected sink_2)
(inspected sinkbasin_0)
(inspected sinkbasin_0)
(inspected sinkbasin_1)
(inspected sinkbasin_1)
(inspected soapbottle_0)
(inspected soapbottle_0)
(open showerdoor_0)
(open showerdoor_1)
(openable showerdoor_0)
(openable showerdoor_1)
(openable toilet_0)
(pickupable candle_0)
(pickupable soapbottle_0)
(pickupable spraybottle_0)
(pickupable towel_0)
(receptacle countertop_0)
(receptacle countertop_1)
(receptacle countertop_2)
(receptacle garbagecan_0)
(receptacle sinkbasin_0)
(receptacle sinkbasin_1)
(receptacle toilet_0)
(receptacle toiletpaperhanger_0)
(receptacle towelholder_0)
(viewing countertop_2)
(viewing faucet_0)
(viewing faucet_1)
(viewing mirror_1)
(viewing sink_1)
(viewing sink_2)
(viewing sinkbasin_0)
(viewing sinkbasin_1)
(viewing soapbottle_0)
)
(:goal
(and
(exists (?o1 - soapbottle) (and (viewing ?o1) (close_to ?o1))))
))