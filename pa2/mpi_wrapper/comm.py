from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        Your implementation should respect the passed reduction operator.
        The required operators for this assignment are MPI.MIN, MPI.SUM,
        and MPI.MAX.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
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
                if dest_rank == root:
                    continue
                self.comm.Send(dest_array, dest=dest_rank)
        else:
            self.comm.Send(src_array, dest=root)
            self.comm.Recv(dest_array, source=root)

        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (nprocs - 1)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
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
