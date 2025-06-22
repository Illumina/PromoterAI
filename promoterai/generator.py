import numpy as np
import tensorflow.keras as tk


def _onehot_encode(seq):
    embed_mtrx = np.zeros((26, 4))
    embed_mtrx[[0, 2, 6, 19], [0, 1, 2, 3]] = 1
    seq_byte = bytearray(seq, encoding='utf8')
    seq_int = np.frombuffer(seq_byte, dtype='int8')
    return embed_mtrx[seq_int - 65]


class SequenceDataGenerator(tk.utils.Sequence):
    def __init__(
            self,
            df_pos,
            fasta,
            bws_fwd,
            bws_rev,
            bw_xforms,
            input_length,
            output_length,
            batch_size,
            shuffle=False
    ):
        self._df_pos = df_pos
        self._idx_smpls = np.arange(len(df_pos))
        self._fasta = fasta
        self._num_bws = len(bws_fwd)
        self._bws_fwd = bws_fwd
        self._bws_rev = bws_rev
        self._bw_xforms = bw_xforms
        self._input_length = input_length
        self._output_length = output_length
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self._idx_smpls) // self._batch_size

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._idx_smpls)

    def __getitem__(self, idx_batch):
        xs = np.zeros((self._batch_size, self._input_length, 4))
        ys = np.zeros((self._batch_size, self._output_length, self._num_bws))

        for i in range(self._batch_size):
            idx_smpl = self._idx_smpls[idx_batch * self._batch_size + i]
            chrom = self._df_pos.iloc[idx_smpl]['chrom']
            pos = self._df_pos.iloc[idx_smpl]['pos'] - 1
            strand = self._df_pos.iloc[idx_smpl]['strand']

            seq = self._fasta[chrom][
                pos - self._input_length // 2:pos + self._input_length // 2
            ].seq.upper()
            if len(seq) < self._input_length:
                continue
            if (strand == 1) or (strand == '+'):
                xs[i] = _onehot_encode(seq)
                for j in range(self._num_bws):
                    ys[i, :, j] = self._bw_xforms[j](self._bws_fwd[j].values(
                        chrom,
                        pos - self._output_length // 2,
                        pos + self._output_length // 2
                    ))
            elif (strand == -1) or (strand == '-'):
                xs[i] = _onehot_encode(seq)[::-1, ::-1]
                for j in range(self._num_bws):
                    ys[i, :, j] = self._bw_xforms[j](self._bws_rev[j].values(
                        chrom,
                        pos - self._output_length // 2,
                        pos + self._output_length // 2
                    ))[::-1]
        return xs, ys


class VariantDataGenerator(tk.utils.Sequence):
    def __init__(
            self,
            df_var,
            fasta,
            input_length,
            batch_size,
            output=None,
            shuffle=False
    ):
        self._df_var = df_var
        self._idx_smpls = np.arange(len(df_var))
        self._fasta = fasta
        self._input_length = input_length
        self._batch_size = batch_size
        self._output = output
        self._shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self._idx_smpls) // self._batch_size

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._idx_smpls)

    def __getitem__(self, idx_batch):
        xs_ref = np.zeros((self._batch_size, self._input_length, 4))
        xs_alt = np.zeros((self._batch_size, self._input_length, 4))
        ys = np.zeros(self._batch_size)

        for i in range(self._batch_size):
            idx_smpl = self._idx_smpls[idx_batch * self._batch_size + i]
            chrom = self._df_var.iloc[idx_smpl]['chrom']
            pos = self._df_var.iloc[idx_smpl]['pos'] - 1
            ref = self._df_var.iloc[idx_smpl]['ref']
            alt = self._df_var.iloc[idx_smpl]['alt']
            strand = self._df_var.iloc[idx_smpl]['strand']

            idx_ref = self._input_length // 2
            seq_ref = self._fasta[chrom][
                pos - self._input_length // 2:pos + self._input_length // 2
            ].seq.upper()
            if len(seq_ref) < self._input_length:
                print(f'Skipping {chrom}:{pos} {ref}>{alt} (pos issue)')
                continue
            if not ref == seq_ref[idx_ref:idx_ref + len(ref)]:
                print(f'Skipping {chrom}:{pos} {ref}>{alt} (ref issue)')
                continue
            if not set(alt).issubset({'A', 'C', 'G', 'T'}):
                print(f'Skipping {chrom}:{pos} {ref}>{alt} (alt issue)')
                continue
            seq_alt = seq_ref[:idx_ref] + alt + seq_ref[idx_ref + len(ref):]
            seq_alt = seq_alt.ljust(len(seq_ref), 'N')[:len(seq_ref)]

            xs_ref[i] = _onehot_encode(seq_ref)
            xs_alt[i] = _onehot_encode(seq_alt)
            if (strand == -1) or (strand == '-'):
                xs_ref[i] = xs_ref[i][::-1, ::-1]
                xs_alt[i] = xs_alt[i][::-1, ::-1]
            if self._output:
                ys[i] = self._df_var.iloc[idx_smpl][self._output]
        return (xs_ref, xs_alt), ys
