"""MPI wrapper for PA3 Part 1.

This wrapper exposes both the buffered NumPy-style collectives that you used
in PA2 (`Allreduce`, `Allgather`, `Reduce_scatter`, `Alltoall`) and the
pickle-based Python-object collectives (`bcast`, `allgather`, `alltoall`, ...).

In Part 1 you will mostly use the pickle-based variants because token routing
in a Mixture-of-Experts produces variable-sized payloads per rank. The buffered
variants are still available if you want to use them in optimized code paths.

If you want to drop in your own implementations of all-reduce / all-to-all from
PA2 (Section 2.1), copy `myAllreduce` and `myAlltoall` from your PA2
`mpi_wrapper/comm.py` into the marked locations below. Doing so is optional but
recommended for the EP implementation; see the bonus rubric in the README.
"""

from mpi4py import MPI
import numpy as np


class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    # ---------- basic info ----------
    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    # ---------- pickle-based (Python object) collectives ----------
    def bcast(self, data, root=0):
        return self.comm.bcast(data, root=root)

    def allgather(self, data):
        return self.comm.allgather(data)

    def alltoall(self, send_data):
        return self.comm.alltoall(send_data)

    def allreduce(self, data, op=MPI.SUM):
        return self.comm.allreduce(data, op=op)

    # ---------- buffered (NumPy) collectives ----------
    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_bytes = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_bytes * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_bytes = src_array.itemsize * src_array.size
        dest_bytes = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_bytes * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_bytes * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()
        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)
        self.comm.Alltoall(src_array, dest_array)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    # ---------- optional: paste your PA2 implementations here ----------
    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """Point-to-point all-reduce implementation copied from PA2."""
        assert src_array.size == dest_array.size

        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()
        root = 0

        if op not in (MPI.SUM, MPI.MIN, MPI.MAX):
            raise ValueError("Unsupported reduction op")

        if nprocs > 0 and (nprocs & (nprocs - 1)) == 0:
            reduced = np.array(src_array, copy=True)
            recv_buf = np.empty_like(src_array)
            mask = 1

            while mask < nprocs:
                peer = rank ^ mask
                self.comm.Sendrecv(reduced, dest=peer, recvbuf=recv_buf, source=peer)
                if op == MPI.SUM:
                    reduced += recv_buf
                elif op == MPI.MIN:
                    np.minimum(reduced, recv_buf, out=reduced)
                else:
                    np.maximum(reduced, recv_buf, out=reduced)
                mask <<= 1

            np.copyto(dest_array, reduced)
        elif rank == root:
            reduced = np.array(src_array, copy=True)
            recv_buf = np.empty_like(src_array)

            for src_rank in range(nprocs):
                if src_rank == root:
                    continue
                self.comm.Recv(recv_buf, source=src_rank)
                if op == MPI.SUM:
                    reduced += recv_buf
                elif op == MPI.MIN:
                    np.minimum(reduced, recv_buf, out=reduced)
                else:
                    np.maximum(reduced, recv_buf, out=reduced)

            np.copyto(dest_array, reduced)
            for dest_rank in range(nprocs):
                if dest_rank != root:
                    self.comm.Send(dest_array, dest=dest_rank)
        else:
            self.comm.Send(src_array, dest=root)
            self.comm.Recv(dest_array, source=root)

        src_bytes = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_bytes * 2 * (nprocs - 1)

    def myAlltoall(self, src_array, dest_array):
        """Point-to-point all-to-all implementation copied from PA2."""
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        assert src_array.size % nprocs == 0
        assert dest_array.size % nprocs == 0

        send_count = src_array.size // nprocs
        recv_count = dest_array.size // nprocs
        assert send_count == recv_count

        src_flat = src_array.reshape(-1)
        dest_flat = dest_array.reshape(-1)

        start = rank * send_count
        end = start + send_count
        dest_flat[start:end] = src_flat[start:end]

        requests = []
        for peer in range(nprocs):
            if peer == rank:
                continue
            recv_start = peer * recv_count
            recv_end = recv_start + recv_count
            requests.append(self.comm.Irecv(dest_flat[recv_start:recv_end], source=peer))

        for peer in range(nprocs):
            if peer == rank:
                continue
            send_start = peer * send_count
            send_end = send_start + send_count
            requests.append(self.comm.Isend(src_flat[send_start:send_end], dest=peer))

        MPI.Request.Waitall(requests)

        send_seg_bytes = src_array.itemsize * send_count
        recv_seg_bytes = dest_array.itemsize * recv_count
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)


# Default global communicator (mirrors the pa2 convention).
mpi = Communicator(MPI.COMM_WORLD)
