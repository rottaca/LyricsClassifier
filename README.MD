# Lyrics Classifier for Song Genre Prediction

This repository contains the source code and the documentation for my project at Stanford University. In this project, I tried to implement a neural network which is capable of predicting a song genre based on a few sentences of song text.

The software is implemented as jupyter notebook and uses Tensorflow and Keras.

The final paper can be found [here](doc/scifile.pdf).

# Results

The per class accuracy is as follows:
```
Per Class accuracy for Country is 48.9 %
Per Class accuracy for Pop is 52.6 %
Per Class accuracy for Rock is 9.9 %
Per Class accuracy for R&B is 8.2 %
Per Class accuracy for Hip-Hop is 85.5 %
Per Class accuracy for Electronic is 09.3 %
Per Class accuracy for Jazz is 0.0 %
```

Here are a few examples of the input and the generated output:
```
Input:
an accident one day that would chill the heart of any man it would teach them not to drink a drop while the steering wheel in their hand this awful accident occurred on the 20th day of may and caused two little children to be sleepin beneath the clay these two little kids walked side by side along the state highway their poor old mother she had died and their daddy had ran away as these two little children walked arm in arm how sad their hearts did feel when around the curb came a speeding car with a drunk
Predicted class 1 with probability 0.4412298798561096: Country
Predicted class 2 with probability 0.1738954484462738: Rock
Predicted class 3 with probability 0.09098871052265167: Metal
Actual class: Country

Input:
of perilous buried to remain among the vanquished there lies a key for which you must retrieve but once within your soul belongs to fire illusive concepts attempts un obtained those who before you became the detained within the gates you seek fading light buried in darkness your sense filled with fright within the gates you will tempt fate you must beware you are the chosen one dark ruler only son the power you posses is now the final chance within the gates you will tempt fate you must beware theme solo theme walk towards the fading light alone wide
Predicted class 1 with probability 0.7165055871009827: Metal
Predicted class 2 with probability 0.11548150330781937: Rock
Predicted class 3 with probability 0.04329933971166611: Electronic
Actual class: Metal

Input:
for the hungry and underprivileged something different from these hollow and grunting niggas this is business strictly step to my business is risky specially when you as bitch as missy back to back lp that sound the same i surround the game with a four pounded brainstorm to make niggas dance in the rain scared to take a chance in the game used to break dance it a shame what money do to a nigga brain if he lose his soul what did a nigga gain doin it doin it i am doin it c o double m o to
Predicted class 1 with probability 0.9738682508468628: Hip-Hop
Predicted class 2 with probability 0.01147608831524849: Metal
Predicted class 3 with probability 0.005403487477451563: Rock
Actual class: Hip-Hop

Input:
me cause one day you will see you will be how you wanna be repeat chorus i remember days when people used to say stop wasting time and find another way they never understood no they never took time that the r o c k was born for the rhymes i kept my mind focused and i went for my direction microphone checkin was my only selection my eyes on the prise i was on the rise everytime you see me now you wanna say hi you dont even know who i am so take all your plans and scram
Predicted class 1 with probability 0.270131379365921: Hip-Hop
Predicted class 2 with probability 0.22825384140014648: Metal
Predicted class 3 with probability 0.15873070061206818: Rock
Actual class: Electronic

Input:
believe me then go ask magic everybody an they momma preachin abstinence these kids aint checkin for absti shit so put a condom in they hand and hope it dont bust another victim of the lust in god we trust what started off as a plan ended up in the plot water cant cool it off cause it too damn hot oh oh it too hot too hot too hot lady too hot gotta run for shelter gotta run for shade it too hot too hot too hot homie too hot gotta make some sense from this mess that we
Predicted class 1 with probability 0.9252456426620483: Hip-Hop
Predicted class 2 with probability 0.02000015787780285: Pop
Predicted class 3 with probability 0.012911594472825527: Rock
Actual class: Hip-Hop

Input:
me mind when i looked behind no bundle could i find upon me stick a wobblin enquiring for the rogue they said me connaught brogue wasnt much in vogue on the rocky road to dublin one two three four five hunt the hare and turn her down the rocky road all the way to dublin whack follol de rah from there i got away me spirits never falling landed on the quay just as the ship was sailing the captain at me roared said that no room had he when i jumped aboard a cabin found for paddy down among
Predicted class 1 with probability 0.3751150369644165: Country
Predicted class 2 with probability 0.18586505949497223: Rock
Predicted class 3 with probability 0.10325896739959717: Pop
Actual class: Pop

Input:
in touch with my heart i wish you didnt always have to pick me apart im a pinch on your arm im in too deep id love to run around with fireworks going off in your sleep you let me loose on myself i couldnt shut you in id love to say im leaving but id hate to pretend i am just a patient you are my replacement why does my life always get me in this situation just a little time id kill to spend it with you baby just a little hope every minute loves to disobey me
Predicted class 1 with probability 0.2732742130756378: Pop
Predicted class 2 with probability 0.20600526034832: R&B
Predicted class 3 with probability 0.16947411000728607: Country
Actual class: Metal

Input:
on the a a a a a a alcohol blame it on the alcohol fades now to the ballers popping bottles with their henny in they cups screaming money aint a thing and if it aint throw it up in the sky hold your drinks up high and to the independent ballers who can buy they own bottles and if you looking like a model when them broke fellas holla tell them bye and hold ya drinks up high you can blame it on the goose goose got you feeling loose loose blame it on petron tron got you in
Predicted class 1 with probability 0.6289089322090149: Hip-Hop
Predicted class 2 with probability 0.11675432324409485: R&B
Predicted class 3 with probability 0.09433288127183914: Country
Actual class: Pop

Input:
becomes dark in the sky everything lost for this human race and the dawn of a new age will rise rays of sunlight now are gone only visions of ice will remain the fallen ones and forgotten souls will rise up over the slain no more hope as we raise our hands to the sky no more dreams as the rivers run dry everything lost all is left astray only sorrow and sadness remains the curtains have dropped on our fallen world and the forces of darkness shall rise why cant you see what has happened to thee can you
Predicted class 1 with probability 0.6968819499015808: Metal
Predicted class 2 with probability 0.12562641501426697: Rock
Predicted class 3 with probability 0.048103153705596924: Pop
Actual class: Metal

Input:
believe what your fantasy let me be your fantasy i know you dream of me every time you fall asleep i can be your fantasy and when you think of me just close your eyes and dream it you and me riding round the city on a shopping spree i can only imagine what you do to me baby come back let me show you things youll never believe im talking bout candle light dinners trips around the world there nothing what i wouldnt give her if she was my girl she never seen a winner take a second and
Predicted class 1 with probability 0.27422651648521423: Pop
Predicted class 2 with probability 0.1738612949848175: R&B
Predicted class 3 with probability 0.1512875258922577: Rock
Actual class: Hip-Hop
```