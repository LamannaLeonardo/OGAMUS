(define (problem ithor-appleinbox)
(:domain ithor)
(:objects
showerdoor_0 - showerdoor
towel_0 - towel
towelholder_0 - towelholder
soapbottle_0 - soapbottle
painting_0 - painting
cabinet_0 - cabinet
cabinet_1 - cabinet
cabinet_2 - cabinet
cabinet_3 - cabinet
cabinet_4 - cabinet
countertop_0 - countertop
countertop_1 - countertop
faucet_0 - faucet
faucet_1 - faucet
faucet_2 - faucet
mirror_0 - mirror
mirror_1 - mirror
soapbar_0 - soapbar
plunger_0 - plunger
toiletpaperhanger_0 - toiletpaperhanger
spraybottle_0 - spraybottle
garbagecan_0 - garbagecan
garbagecan_1 - garbagecan
toilet_0 - toilet
toilet_1 - toilet
sinkbasin_0 - sinkbasin
sinkbasin_1 - sinkbasin
sinkbasin_2 - sinkbasin
sink_0 - sink
sink_1 - sink
sink_2 - sink
candle_0 - candle
handtowel_0 - handtowel
handtowelholder_0 - handtowelholder
scrubbrush_0 - scrubbrush
)
(:init
(close_to cabinet_0)
(close_to cabinet_0)
(close_to cabinet_1)
(close_to cabinet_1)
(close_to countertop_0)
(close_to countertop_0)
(close_to countertop_1)
(close_to countertop_1)
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
(close_to sinkbasin_2)
(close_to sinkbasin_2)
(close_to soapbar_0)
(close_to soapbar_0)
(close_to soapbottle_0)
(close_to soapbottle_0)
(close_to towel_0)
(close_to towel_0)
(close_to towelholder_0)
(close_to towelholder_0)
(discovered cabinet_0)
(discovered cabinet_1)
(discovered cabinet_2)
(discovered cabinet_3)
(discovered cabinet_4)
(discovered candle_0)
(discovered countertop_0)
(discovered countertop_1)
(discovered faucet_0)
(discovered faucet_1)
(discovered faucet_2)
(discovered garbagecan_0)
(discovered garbagecan_1)
(discovered handtowel_0)
(discovered handtowelholder_0)
(discovered mirror_0)
(discovered mirror_1)
(discovered painting_0)
(discovered plunger_0)
(discovered scrubbrush_0)
(discovered showerdoor_0)
(discovered sink_0)
(discovered sink_1)
(discovered sink_2)
(discovered sinkbasin_0)
(discovered sinkbasin_1)
(discovered sinkbasin_2)
(discovered soapbar_0)
(discovered soapbottle_0)
(discovered spraybottle_0)
(discovered toilet_0)
(discovered toilet_1)
(discovered toiletpaperhanger_0)
(discovered towel_0)
(discovered towelholder_0)
(hand_free )
(inspected countertop_1)
(inspected countertop_1)
(inspected mirror_1)
(inspected mirror_1)
(inspected sink_1)
(inspected sink_1)
(inspected sink_2)
(inspected sink_2)
(inspected sinkbasin_2)
(inspected sinkbasin_2)
(inspected soapbar_0)
(inspected soapbar_0)
(inspected soapbottle_0)
(inspected soapbottle_0)
(inspected towel_0)
(inspected towelholder_0)
(open showerdoor_0)
(openable cabinet_0)
(openable cabinet_1)
(openable cabinet_2)
(openable cabinet_3)
(openable cabinet_4)
(openable showerdoor_0)
(openable toilet_0)
(openable toilet_1)
(pickupable candle_0)
(pickupable handtowel_0)
(pickupable plunger_0)
(pickupable scrubbrush_0)
(pickupable soapbar_0)
(pickupable soapbottle_0)
(pickupable spraybottle_0)
(pickupable towel_0)
(receptacle cabinet_0)
(receptacle cabinet_1)
(receptacle cabinet_2)
(receptacle cabinet_3)
(receptacle cabinet_4)
(receptacle countertop_0)
(receptacle countertop_1)
(receptacle garbagecan_0)
(receptacle garbagecan_1)
(receptacle handtowelholder_0)
(receptacle sinkbasin_0)
(receptacle sinkbasin_1)
(receptacle sinkbasin_2)
(receptacle toilet_0)
(receptacle toilet_1)
(receptacle toiletpaperhanger_0)
(receptacle towelholder_0)
(viewing candle_0)
(viewing countertop_1)
(viewing faucet_1)
(viewing faucet_2)
(viewing mirror_1)
(viewing sink_1)
(viewing sink_2)
(viewing sinkbasin_1)
(viewing sinkbasin_2)
(viewing soapbar_0)
(viewing soapbottle_0)
(viewing spraybottle_0)
(viewing toilet_1)
)
(:goal
(and
(exists (?o1 - soapbottle) (and (viewing ?o1) (close_to ?o1))))
))