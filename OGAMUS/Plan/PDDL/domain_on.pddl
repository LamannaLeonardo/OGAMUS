(define (domain ithor)
(:requirements :typing)
(:types
alarmclock
aluminumfoil
apple
armchair
baseballbat
basketball
bathtub
bathtubbasin
bed
blinds
book
boots
bottle
bowl
box
bread
butterknife
cd
cabinet
candle
cellphone
chair
cloth
coffeemachine
coffeetable
countertop
creditcard
cup
curtains
desk
desklamp
desktop
diningtable
dishsponge
dogbed
drawer
dresser
dumbbell
egg
faucet
floor
floorlamp
footstool
fork
fridge
garbagebag
garbagecan
handtowel
handtowelholder
houseplant
kettle
keychain
knife
ladle
laptop
laundryhamper
lettuce
lightswitch
microwave
mirror
mug
newspaper
ottoman
painting
pan
papertowelroll
pen
pencil
peppershaker
pillow
plate
plunger
poster
pot
potato
remotecontrol
roomdecor
safe
saltshaker
scrubbrush
shelf
shelvingunit
showercurtain
showerdoor
showerglass
showerhead
sidetable
sink
sinkbasin
soapbar
soapbottle
sofa
spatula
spoon
spraybottle
statue
stool
stoveburner
stoveknob
tvstand
tabletopdecor
teddybear
television
tennisracket
tissuebox
toaster
toilet
toiletpaper
toiletpaperhanger
tomato
towel
towelholder
vacuumcleaner
vase
watch
wateringcan
window
winebottle
receptacle - object
pickupable - object
openable - object
)

(:predicates
        (hand_free)
		(holding ?o - object)
		(on ?o1 ?o2 - object)
		(close_to ?o - object)
		(open ?o - object)
		(discovered ?o - object)
		(openable ?o - object)
		(pickupable ?o - object)
		(receptacle ?o - object)
		(viewing ?o - object)
		(inspected ?o - object)
)

(:action pickup
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (pickupable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		              )
		:effect
		        (and
		            (not (hand_free))
		            (holding ?param_1)
		         )
)


(:action putinto
		:parameters (?param_1 ?param_2 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (discovered ?param_2)
		                    (holding ?param_1)
		                    (close_to ?param_2)
		                    (open ?param_2)
		                    (receptacle ?param_2)
		                    (viewing ?param_2)
		                    (inspected ?param_1)
		                    (inspected ?param_2)
		              )
		:effect
		        (and
		            (hand_free)
		            (not (holding ?param_1))
		            (on ?param_1 ?param_2)
		         )
)

(:action puton
		:parameters (?param_1 ?param_2 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (discovered ?param_2)
		                    (holding ?param_1)
		                    (close_to ?param_2)
		                    (receptacle ?param_2)
		                    (viewing ?param_2)
		                    (inspected ?param_2)
		              )
		:effect
		        (and
		            (hand_free)
		            (not (holding ?param_1))
		            (on ?param_1 ?param_2)
		         )
)

(:action open
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (openable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		              )
		:effect
		        (and
		            (open ?param_1)
		         )
)

(:action close
		:parameters (?param_1 - object)
		:precondition (and
		                    (discovered ?param_1)
		                    (close_to ?param_1)
		                    (hand_free)
		                    (openable ?param_1)
		                    (viewing ?param_1)
		                    (inspected ?param_1)
		              )
		:effect
		        (and
		            (not (open ?param_1))
		         )
)

(:action get_close_and_look_at_receptacle
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (not (hand_free))
		                (receptacle ?param_1)
		                (inspected ?param_1)
		                (or
		                (not (close_to ?param_1))
		                (not (viewing ?param_1))
		                )
		              )
		:effect
		        (and
		            (close_to ?param_1)
		            (viewing ?param_1)
		            (forall (?x - object) (not (viewing ?x)))
		         )
)




(:action get_close_and_look_at_pickupable
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (pickupable ?param_1)
		                (inspected ?param_1)
		                (or
		                (not (close_to ?param_1))
		                (not (viewing ?param_1))
		                )
		              )
		:effect
		        (and
		            (close_to ?param_1)
		            (viewing ?param_1)
		            (forall (?x - object) (not (viewing ?x)))
		         )
)


(:action inspect
		:parameters (?param_1 - object)
		:precondition (and
		                (discovered ?param_1)
		                (not (inspected ?param_1))
		              )
		:effect
		        (and
		            (inspected ?param_1)
		            (close_to ?param_1)
		            (viewing ?param_1)
		            (forall (?x - object) (not (viewing ?x)))
		         )
)


(:action stop
		:parameters ()
		:precondition (and)
		:effect (and)
)

)

