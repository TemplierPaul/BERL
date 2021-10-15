from berl import *

comm = MPI.COMM_WORLD
size = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if rank == 0:
        pb = run_xp(args)
        pb.eval_hof()
        # pb.plot()
        # pb.render(n=5)
        pb.close_MPI()
    else:
        args = load_preset(args)
        net = NETWORKS[args.net.lower()](args.env)
        cfg = args.__dict__
        s = Secondary(net, cfg)
        s.run()
