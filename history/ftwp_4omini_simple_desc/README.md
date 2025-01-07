## 2024.1.7

We reconstructed the prompt for description simplification.

Old style: 

> Reform the environment description using format: Room[Furniture[Object]]

> Example: Bedroom[wardrobe, chest of drawers[black sock], desk[pen, eraser], patio door]

New style:

> Reform the environment description using format: Room[Furniture(status)[Objects]]

> Example: Bedroom[wardrobe(closed), chest of drawers(opened)[black sock], desk[pen, eraser], patio door(opened, west)]


## Results Summarization

* BEST EPOCH: 2
* TEST ON TEMP SET (20 games): 0.6209150326797386
* TEST ON REAL SET (10 games): 0.3230769230769231
