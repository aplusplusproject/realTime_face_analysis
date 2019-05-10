import os
import argparse

def rename(dir, name):
	folder = os.path.join(dir,name)
	os.chdir(os.path.join(os.getcwd(), folder))
	cntr = 0
	for filename in os.listdir("./"):
		cntr += 1
		number = str(cntr)
		while(len(number) != 4): number = "0" + number
		newname = name + "_" + number + filename[-4:]
		os.rename(filename, newname)

def main():
    parser = argparse.ArgumentParser()
    #input directory and name
    parser.add_argument("--dir", type=str, default="./output/")
    parser.add_argument("--name", type=str, default="Mark_Christopher_Uy")
    args = parser.parse_args()

    rename(args.dir, args.name)

if __name__=="__main__":
    main()