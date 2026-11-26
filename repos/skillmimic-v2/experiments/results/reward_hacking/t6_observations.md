# Test 6: what the policies do (filled in by hand while watching test_6.sh)

For each checkpoint, one or two lines per question. "Don't know" is a fine answer.

1. Does it pick the object up? With which hand(s)? Does the grip look like the demonstration?
2. After picking it up, does it carry the object along the demonstrated path, or stop / drop / leave it somewhere?
3. Does the body keep copying the demonstration (walking, turning) even when the object is no longer in the hand?
4. Anything odd: object pushed with the arm or body instead of the hand, object stuck to the hand or launched,
   hand passing through the object or furniture, sliding feet, falling on purpose, standing still.
5. When does it go wrong (roughly which part of the clip: reach / pick up / lift / carry / put down)?

## place_book epoch 1000 (lift 100%, falls 88% in the 8-env curve)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Lifts on time (frame 80, height 0.40 m vs demo 0.45 m) and carries along the demo path.
- Drifts off the path from about frame 94 (58 cm off by frame 128). Falls at the end of the clip (frame 148).


## place_book epoch 2000 (lift 0%)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Hand reaches the book (frames 44-72, wrist 10-14 cm away) but never lifts it; the book slides 22 cm on the table.
- Goes wrong at pick-up. The person falls around frames 100-114.


## place_book epoch 3000 (lift 38%)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Hand stays next to the book the whole clip (14 cm) but drags it 40 cm instead of lifting; only 17 cm high by the end.
- Misses the lift timing (demo lifts from frame 57). Does not fall in the clip.


## drink_cup epoch 2100 (lift 0%, one checkpoint before grasping appears)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Hand never goes to the cup (19-59 cm away); the cup never moves. The arms follow the demo motion in the air.
- Same behaviour as the test 3 cheat "body perfect, object untouched".


## drink_cup epoch 2200 (lift 88%)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Lifts on time and holds the cup steadily (wrist 15 cm, height 0.75 m).
- Puts it down about 20 frames late (demo puts down at frames 104-121).


## drink_cup epoch 3000 (lift 75%)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Lifts on time and holds steadily, puts it down about 35 frames late. No fall.


## place_book epoch 2000 from frame 30 (START=30; test 4: lifts 100% from here, 0% from frame 2)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Same checkpoint as above, but lifts on time (frame 76, 0.38 m) and follows the demo path within 10 cm.
- Falls after frame 114 and drops the book.


## drink_cup epoch 2000 from frame 30 (START=30; test 4: lifts 100% from here, 0% from frame 2)

(from the recorded viewer rollout, not from watching; add what you saw below)
- Lifts the cup (it never lifts from frame 2), holds it, puts it down about 30 frames late.

