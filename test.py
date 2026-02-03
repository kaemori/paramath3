import random

print("The files:")
print(
    " ".join(
        [
            (
                "█" * random.randint(5, 20)
                if random.random() > 0.2
                else random.choice(
                    ["the", "a", "an", "some", "any", "this", "that", "these", "those"]
                )
            )
            for _ in range(30)
        ]
    )
)
