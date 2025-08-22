1) Run the SFT classifier with 512, 1024, 2048, 4096 and full 
    a) record their accuracy, think proportion, answer proportion and length
2) Run the GRPO with 512, 1024, 2048, 4096 and full
    a) record their accuracy, think proportion, answer proportion and length
3) Run the GRPO with shorten answer reward with 512, 1024, 2048, 4096 and full
    a) record their accuracy, think proportion, answer proportion and length


Actually, do this whole cycle with each weight. We should be getting the sample n from an env variable

Questions?
    - How many mor epochs (if any) do we need to match the SFT performance with the GRPO?
    - What happens if we decrease the number of generations per sample?