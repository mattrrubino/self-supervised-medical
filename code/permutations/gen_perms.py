from pathlib import Path
import numpy as np
import math

if __name__ == "__main__":
    num_patches = int(input("How many patches are there? "))
    num_perms = int(input("How many permutations do you want? "))

    max_perms = math.factorial(num_patches)
    if num_perms > max_perms:
        print("Too many perms requested, only", max_perms, "possible for", num_patches, "patches.")
        exit()


    perms = []

    while len(perms) < num_perms:
        pp = np.random.permutation(num_patches)
        # print(pp)
        
        duplicate_perm = False
        for ref in perms:
            if (pp==ref).all():
                duplicate_perm = True
                break
        if duplicate_perm:
            continue
        else:
            perms.append(pp)

    perms = np.stack(perms)


    output_name = "permutations_"+ str(num_perms) +"_" + str(num_patches) + ".npy"

    permutation_path = str(
        Path(__file__).parent / output_name
    )

    print(permutation_path)

    with open(permutation_path, 'wb') as f:
        np.save(f, perms)