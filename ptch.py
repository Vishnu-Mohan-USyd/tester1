for b0 in range(0, B_tot, batch_cap):
    b1  = min(b0 + batch_cap, B_tot)
    sel = slice(b0, b1)
    bs  = sel.stop - sel.start         # <— the critical line
    net.reset_state(bs)

    for t in range(T):
        net.update_all_layers_batch(xA[sel, t], xV[sel, t])

    # … decoding logic unchanged …
