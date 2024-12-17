from ubuntu

RUN apt update && apt upgrade -y && apt install -y curl build-essential libffi-dev libffi8 libgmp-dev libgmp10 libncurses-dev libncurses5 libtinfo5 pkg-config

#RUN curl -sSL https://get.haskellstack.org/ | sh
#RUN stack upgrade
RUN curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
#RUN mkdir /workdir
#COPY ./newsynth-0.4.1.0 /workdir
#RUN stack build /workdir/newsynth/programs/gridsynth.hs

ENV PATH="/root/.ghcup/bin:/root/.cabal.bin:$PATH"
#RUN curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | BOOTSTRAP_HASAKELL_MINIMAL=1 sh
#ENTRYPOINT ["source /root/.ghcup/env && cabal install newsynth"]
RUN cabal install newsynth
RUN cp $(readlink -f /root/.cabal/bin/gridsynth) /root/gridsynth
