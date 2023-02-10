
        
def run_validation(pre, dataset, dataloader, model, epoch, label, device, writer):
    
    # VALIDATION
    print(f"VALTEST | {label} of epoch {epoch} ===")
    model.eval()
    with torch.no_grad():
        info = dataset.data
        sample_id = []
        affinity_pred_list = []
        result = []
        length = 0
        loss = 0
        recto_rate = 0
            
            
        # sample scope
        print(f"VALTEST: {label} | Compute affinity")
        for data in tqdm(dataloader):
            sample_id.extend(data.sample_id)
            data = data.to(device)
            _, affinity_pred = model(data)
            del _
            affinity_pred_list.append(affinity_pred.detach().cpu())
        # end of sample scope
            
        # write to DataFrame
        info["dataset_sample_id"] = sample_id
        info["affinity_pred"] = torch.cat(affinity_pred_list).tolist()
            
        print(f"VALTEST: {label} Compute loss and recto_ratio")
        for session in info.session_au.unique():
            df = info[validation_info.session_au==session]
            affinity_true = torch.tensor(df.value.values).to(device)
            affinity_pred = torch.tensor(df.affinity_pred.values).to(device)
            loss, recto_rate, num_pairs = pairwiseloss(pred=affinity_pred, true=affinity_true)
            loss = loss.detach().cpu()
            recto_rate = recto_rate.detach().cpu()
            session_len= len(df)
                
            length += np.sqrt(num_pairs)
            loss += loss * np.sqrt(num_pairs)
            recto_rate += recto_rate * np.sqrt(num_pairs)
                
            result.append([session, session_len, num_pairs, loss.item(), recto_rate.item()])
            # end of sample scope: ratio and loss calculate
            
            
    # save result as .csv
    result = pd.DataFrame(result, columns=["session", "length", "num_pairs", "loss", "recto_rate"])
    savepath = f"{pre}/results/{label}_result_{epoch}.csv"
    result.to_csv(savepath)
    info.to_csv(f"{pre}/results/{label}_info_{epoch}.csv")
    print(f"VALTEST: {label} | Save result of epoch {epoch} to {savepath}.")
            
            
    # save result to tensorboard
    loss /= length
    recto_rate /= length
    writer.add_scalar(f'rank_loss.by_epoch/iid_validation', loss.item(), epoch)
    writer.add_scalar(f'recto_rate.by_epoch/iid_validation', recto_rate.item(), epoch)
    logging.info(f"epoch {epoch} - {label} | rank_loss {loss.item()}, averaged_recto_rate {recto_rate.item()}")